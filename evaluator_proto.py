import sys
import logging
import copy
import torch
import torch.nn.functional as F
from utils import factory
from utils.data_manager import DataManager
from torch.utils.data import DataLoader
from utils.toolkit import count_parameters
import os
import numpy as np


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove a leading 'module.' prefix (DataParallel) if present."""
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}

def _infer_num_classes_from_state_dict(state_dict: dict) -> int:
    """Infer #classes from common classifier weight keys."""
    cand_keys = [
        "fc.weight",
        "classifier.weight",
        "linear.weight",
        "oldfc.weight",  # foster
    ]
    for k in cand_keys:
        if k in state_dict and hasattr(state_dict[k], "shape"):
            return int(state_dict[k].shape[0])

    # Fallback: find the first 2D tensor whose name ends with '.weight' and contains 'fc'
    for k, v in state_dict.items():
        if ("fc" in k) and k.endswith("weight") and hasattr(v, "shape") and len(v.shape) == 2:
            return int(v.shape[0])
    raise KeyError(
        "Could not infer #classes from checkpoint. Expected keys like 'fc.weight' or 'oldfc.weight'."
    )

def _prepare_network_for_phase(model, data_manager: DataManager, phase_id: int):
    """Build the same dynamic architecture as training, up to `phase_id`.

    - For DER/TagFex/FOSTER etc, update_fc is called once per task.
    - For simple backbones, one update_fc(total_classes) is enough.
    """
    if not hasattr(model, "_network") or not hasattr(model._network, "update_fc"):
        return

    # total classes at each task boundary
    totals = []
    known = 0
    for t in range(phase_id + 1):
        known += data_manager.get_task_size(t)
        totals.append(known)

    # Methods with dynamic expansion typically keep a ModuleList called "convnets".
    # For those, we MUST call update_fc(task_total) per task.
    if hasattr(model._network, "convnets"):
        for total in totals:
            model._network.update_fc(total)
    else:
        model._network.update_fc(totals[-1])

    # Mirror training-time bookkeeping (helps for reporting).
    model._cur_task = phase_id
    model._total_classes = totals[-1]
    model._known_classes = totals[-1] - data_manager.get_task_size(phase_id)

@torch.no_grad()
def _build_prototypes(
    model,
    data_manager: DataManager,
    num_classes: int,
    n_per_class: int,
    batch_size: int,
    num_workers: int,
    l2_normalize: bool,
):
    """Compute per-class prototypes as mean feature vectors using up to N samples per class."""
    # Use train split but test transforms (PyCIL does this when computing class means).
    proto_dataset = data_manager.get_dataset(
        np.arange(0, num_classes), source="train", mode="test"
    )
    proto_loader = DataLoader(
        proto_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Determine the feature extractor pointer
    net = model._network.module if hasattr(model._network, "module") else model._network

    feats_by_class = [[] for _ in range(num_classes)]

    for _, (_, inputs, targets) in enumerate(proto_loader):
        inputs = inputs.to(model._device, non_blocking=True)
        feats = net.extract_vector(inputs)
        if l2_normalize:
            feats = F.normalize(feats, dim=1)

        for f, t in zip(feats, targets):
            c = int(t)
            if c < 0 or c >= num_classes:
                continue
            if len(feats_by_class[c]) < n_per_class:
                feats_by_class[c].append(f.detach().cpu())

        if all(len(v) >= n_per_class for v in feats_by_class):
            break

    # Mean per class
    feat_dim = feats_by_class[0][0].numel() if len(feats_by_class[0]) > 0 else net.feature_dim
    protos = []
    used = []
    for c in range(num_classes):
        if len(feats_by_class[c]) == 0:
            proto = torch.zeros(feat_dim)
        else:
            proto = torch.stack(feats_by_class[c], dim=0).mean(dim=0)
        if l2_normalize:
            proto = F.normalize(proto, dim=0)
        protos.append(proto)
        used.append(len(feats_by_class[c]))

    protos = torch.stack(protos, dim=0).to(model._device)  # [C, D]
    return protos, used

@torch.no_grad()
def _prototype_predict(
    model,
    loader: DataLoader,
    prototypes: torch.Tensor,
    topk: int,
    metric: str = "sqeuclidean",
    l2_normalize: bool = True,
):
    """Return (y_pred [N, topk], y_true [N]) by nearest-prototype."""
    net = model._network.module if hasattr(model._network, "module") else model._network
    net.eval()

    y_pred, y_true = [], []
    for _, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(model._device, non_blocking=True)
        feats = net.extract_vector(inputs)
        if l2_normalize:
            feats = F.normalize(feats, dim=1)

        if metric.lower() in {"cos", "cosine"}:
            scores = feats @ prototypes.t()  # larger is closer
            pred = torch.topk(scores, k=topk, dim=1, largest=True, sorted=True).indices
        else:
            # squared euclidean distance: smaller is closer
            x2 = (feats ** 2).sum(dim=1, keepdim=True)  # [B,1]
            p2 = (prototypes ** 2).sum(dim=1).unsqueeze(0)  # [1,C]
            dists = x2 + p2 - 2.0 * (feats @ prototypes.t())
            pred = torch.topk(-dists, k=topk, dim=1, largest=True, sorted=True).indices

        y_pred.append(pred.cpu().numpy())
        y_true.append(targets.cpu().numpy())

    return np.concatenate(y_pred), np.concatenate(y_true)


def _evaluate_topk(y_pred: np.ndarray, y_true: np.ndarray, nb_old: int, increment: int, topk: int):
    """Standalone version of PyCIL's BaseLearner._evaluate (avoids signature drift)."""
    y_true = np.asarray(y_true)
    top1 = y_pred[:, 0]

    grouped = {}
    grouped["total"] = np.around((top1 == y_true).sum() * 100.0 / len(y_true), decimals=2)

    # Grouped accuracy by class ranges
    max_c = int(np.max(y_true))
    for class_id in range(0, max_c + 1, increment):
        idxes = np.where((y_true >= class_id) & (y_true < class_id + increment))[0]
        if len(idxes) == 0:
            acc = 0.0
        else:
            acc = np.around((top1[idxes] == y_true[idxes]).sum() * 100.0 / len(idxes), decimals=2)
        label = f"{class_id:02d}-{class_id + increment - 1:02d}"
        grouped[label] = acc

    # Old/New
    idx_old = np.where(y_true < nb_old)[0]
    grouped["old"] = 0.0 if len(idx_old) == 0 else np.around((top1[idx_old] == y_true[idx_old]).sum() * 100.0 / len(idx_old), decimals=2)
    idx_new = np.where(y_true >= nb_old)[0]
    grouped["new"] = 0.0 if len(idx_new) == 0 else np.around((top1[idx_new] == y_true[idx_new]).sum() * 100.0 / len(idx_new), decimals=2)

    ret = {"grouped": grouped, "top1": grouped["total"]}
    # topk
    topk_acc = (y_pred.T[:topk] == np.tile(y_true, (topk, 1))).sum() * 100.0 / len(y_true)
    ret[f"top{topk}"] = np.around(topk_acc, decimals=2)
    return ret



def evaluate(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _eval(args)

def _add_retain_forget_metrics(accy: dict, y_pred: np.ndarray, y_true: np.ndarray, forget_set: set):
    y_true = np.asarray(y_true)
    top1 = y_pred[:, 0]

    if len(forget_set) > 0:
        mask_forget = np.isin(y_true, list(forget_set))
    else:
        mask_forget = np.zeros_like(y_true, dtype=bool)
    mask_retain = ~mask_forget

    def _top1(mask):
        if not mask.any():
            return None
        return np.around((top1[mask] == y_true[mask]).sum() * 100.0 / mask.sum(), decimals=2)

    retain_top1 = _top1(mask_retain)
    forget_top1 = _top1(mask_forget)

    accy["retain_top1"] = retain_top1
    accy["forget_top1"] = forget_top1
    accy["num_retain_samples"] = int(mask_retain.sum())
    accy["num_forget_samples"] = int(mask_forget.sum())

    if forget_top1 is not None:
        forget_err = 100.0 - float(forget_top1)
    else:
        forget_err = None
    accy["forget_err"] = forget_err

    if (retain_top1 is not None) and (forget_err is not None):
        denom = retain_top1 + forget_err
        hmean = 0.0 if denom <= 0 else (2.0 * retain_top1 * forget_err / denom)
    else:
        hmean = None
    accy["hmean"] = hmean

    return accy

def _eval(args):

    #=== log name の決定 ===#
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    log = "baseline" if "log" not in args else args["log"]
    # logs_name = "logs/{}/{}/{}/{}/{}".format(log, args["model_name"],args["dataset"], init_cls, args['increment'])

    #=== log の設定 ===#
    log_dir = "logs/{}/{}/{}/{}/{}/{}_{}_{}/".format(
        args["model_name"],
        log,
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"], args["seed"], args["convnet_type"],
    )
    
    #=== log デイレク取りを作成 ===#
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=log_dir + "exp.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    #=== seed 値の固定 ===#
    _set_random()
    _set_device(args)
    print_args(args)

    #=== data manager の設定 ===#
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["aug"] if "aug" in args else 1
    )

    #=== model の作成 ===#
    model = factory.get_model(args["model_name"], args)

    #=== 学習済みパラメータの読み込み ===#
    ckpt_dir  = build_ckpt_dir(args)
    ckpt_path = f"{ckpt_dir}/phase_{args['phase_id']}.pkl"

    ckpt = torch.load(ckpt_path, map_location=model._device)

    # --- state_dict を取り出す（save_checkpoint 実装差分に対応） ---
    state_dict = (
        ckpt.get("model_state_dict", None)
        if isinstance(ckpt, dict)
        else None
    )
    if state_dict is None and isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt.get("net", None))
    if state_dict is None:
        raise KeyError("Checkpoint does not contain a model state dict.")
    state_dict = _strip_module_prefix(state_dict)

    # --- phase_id に応じて動的にネットワーク構造を組み立てる ---
    _prepare_network_for_phase(model, data_manager, int(args["phase_id"]))

    # --- checkpoint と config の整合性チェック（参考情報） ---
    try:
        n_cls_ckpt = _infer_num_classes_from_state_dict(state_dict)
        if n_cls_ckpt != model._total_classes:
            logging.warning(
                f"[ProtoEval] #classes mismatch: checkpoint={n_cls_ckpt}, config(total@phase)={model._total_classes}. "
                "phase_id / init_cls / increment / dataset が一致しているか確認してください。"
            )
            # ここでは強制的に合わせない（DER/TagFex の task_sizes が崩れる）
    except Exception as e:
        logging.warning(f"[ProtoEval] Could not infer #classes from checkpoint: {e}")


    # --- 重みをロード ---
    model._network.to(model._device)
    try:
        model._network.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.warning(f"[ProtoEval] strict=True で load_state_dict に失敗: {e}")
        logging.warning("[ProtoEval] strict=False で再試行します（不足/余剰キーを無視）")
        model._network.load_state_dict(state_dict, strict=False)

    # (任意) forget_classes を保存しているチェックポイントなら復元
    if isinstance(ckpt, dict) and "forget_classes" in ckpt:
        model.forget_classes = ckpt["forget_classes"]

    # --- dataloader 構築 ---
    num_classes = int(model._total_classes)
    eval_bs = int(args.get("batch_size", 128))
    num_workers = int(args.get("num_workers", 4))

    test_dataset = data_manager.get_dataset(
        np.arange(0, num_classes), source="test", mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # train 精度も見たい場合に備えて（テスト変換で評価）
    train_dataset_for_eval = data_manager.get_dataset(
        np.arange(0, num_classes), source="train", mode="test"
    )
    train_loader_for_eval = DataLoader(
        train_dataset_for_eval,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- prototype 計算 ---
    proto_n = int(args.get("proto_n", args.get("proto_num", 20)))
    proto_bs = int(args.get("proto_batch_size", eval_bs))
    proto_metric = str(args.get("proto_metric", "sqeuclidean"))
    proto_l2norm = bool(args.get("proto_l2norm", True))
    topk = int(getattr(model, "topk", 5))

    prototypes, used_per_class = _build_prototypes(
        model=model,
        data_manager=data_manager,
        num_classes=num_classes,
        n_per_class=proto_n,
        batch_size=proto_bs,
        num_workers=num_workers,
        l2_normalize=proto_l2norm,
    )

    logging.info(
        "[ProtoEval] Prototype samples per class: min={}, max={}, target={}"
        .format(int(np.min(used_per_class)), int(np.max(used_per_class)), proto_n)
    )

    # --- prototype 分類で評価（test / train） ---
    y_pred_test, y_true_test = _prototype_predict(
        model=model,
        loader=test_loader,
        prototypes=prototypes,
        topk=topk,
        metric=proto_metric,
        l2_normalize=proto_l2norm,
    )
    proto_accy_test = _evaluate_topk(
        y_pred=y_pred_test,
        y_true=y_true_test,
        nb_old=int(model._known_classes),
        increment=int(args.get("increment", 10)),
        topk=topk,
    )

    y_pred_train, y_true_train = _prototype_predict(
        model=model,
        loader=train_loader_for_eval,
        prototypes=prototypes,
        topk=topk,
        metric=proto_metric,
        l2_normalize=proto_l2norm,
    )
    proto_accy_train = _evaluate_topk(
        y_pred=y_pred_train,
        y_true=y_true_train,
        nb_old=int(model._known_classes),
        increment=int(args.get("increment", 10)),
        topk=topk,
    )

    logging.info("PROTO (train): {}".format(proto_accy_train["grouped"]))
    logging.info("PROTO (test):  {}".format(proto_accy_test["grouped"]))
    logging.info("PROTO train top1: {}".format(proto_accy_train["top1"]))
    logging.info("PROTO test  top1: {}".format(proto_accy_test["top1"]))
    logging.info("PROTO train top{}: {}".format(topk, proto_accy_train[f"top{topk}"]))
    logging.info("PROTO test  top{}: {}".format(topk, proto_accy_test[f"top{topk}"]))

    # --- forget_classes を参照して retain/forget 指標を追加 ---
    model.forget_classes = [0,1,10,11,20,21,30,31,40,41,50,51,60,61,70,71,80,81]
    forget_set = {c for c in getattr(model, "forget_classes", []) if 0 <= c < num_classes}

    proto_accy_test = _add_retain_forget_metrics(proto_accy_test, y_pred_test, y_true_test, forget_set)
    proto_accy_train = _add_retain_forget_metrics(proto_accy_train, y_pred_train, y_true_train, forget_set)

    logging.info(
        "[PROTO][Train] retain_top1={}, forget_top1={}, forget_err={}, hmean={}, n_retain={}, n_forget={}".format(
            proto_accy_train["retain_top1"], proto_accy_train["forget_top1"],
            proto_accy_train["forget_err"], proto_accy_train["hmean"],
            proto_accy_train["num_retain_samples"], proto_accy_train["num_forget_samples"]
        )
    )
    logging.info(
        "[PROTO][Test ] retain_top1={}, forget_top1={}, forget_err={}, hmean={}, n_retain={}, n_forget={}".format(
            proto_accy_test["retain_top1"], proto_accy_test["forget_top1"],
            proto_accy_test["forget_err"], proto_accy_test["hmean"],
            proto_accy_test["num_retain_samples"], proto_accy_test["num_forget_samples"]
        )
    )







def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus

def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def build_ckpt_dir(args):
    
    """trainer / BaseLearner.save_checkpoint と同じ規則で checkpoint ディレクトリを作る"""
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    log = "baseline" if "log" not in args else args["log"]
    
    ckpt_dir = "logs/{}/{}/{}/{}/{}/{}_{}_{}/".format(
        args["model_name"],
        log,
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"], args["seed"], args["convnet_type"],
    )
    return ckpt_dir






