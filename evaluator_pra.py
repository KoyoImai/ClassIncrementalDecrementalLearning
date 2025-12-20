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


def evaluate(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _eval(args)


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

    # state_dict 取り出しを頑丈に
    state_dict = ckpt.get("state_dict", None)
    if state_dict is None:
        state_dict = ckpt.get("model_state_dict", None)
    if state_dict is None:
        raise RuntimeError("No state_dict found in checkpoint")

    # fc 出力次元
    if "fc.weight" not in state_dict:
        raise RuntimeError("fc.weight not found in checkpoint state_dict")
    out_dim = state_dict["fc.weight"].shape[0]

    phase_id = int(args["phase_id"])

    # --- DERなら：タスクごとの累積クラス数で update_fc を繰り返す ---
    if hasattr(model._network, "convnets"):
        total = 0
        for t in range(phase_id + 1):
            total += data_manager.get_task_size(t)   # ←ここが肝
            model._network.update_fc(total)

        if total != out_dim:
            logging.warning(f"[PRA] total classes from DataManager ({total}) != fc out_dim in ckpt ({out_dim})")
    else:
        # 単一-backbone系
        model._network.update_fc(out_dim)

    # ここでロード（strict=TrueでOKになる）
    model._network.load_state_dict(state_dict, strict=True)
    model.forget_classes = ckpt["forget_classes"]

    # model.forget_classes = [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71, 80, 81]
    # model.forget_classes = [0, 1]
    # model.forget_classes = [0, 1, 50, 51, 60, 61, 70, 71, 80, 81]


    #=== プロトタイプ再学習攻撃を行う前の精度評価 ===#
    # phase_id を _cur_task として使用
    model._cur_task = int(args.get("phase_id", -1))

    # fc 出力次元 = 現時点で扱うクラス数
    model._total_classes = int(out_dim)
    # PyCIL 系は after_task 後に known == total になるのが通常
    model._known_classes = int(out_dim)

    # 訓練データセットの作成
    train_dataset = data_manager.get_dataset(
        np.arange(0, model._total_classes),
        source="train",
        mode="test",
    )

    # 訓練用ローダーの作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.get("batch_size", 128),
        shuffle=False,
        num_workers=args.get("num_workers", 4),
        pin_memory=True,
    )

    # テストデータセットの作成
    test_dataset = data_manager.get_dataset(
        np.arange(0, model._total_classes),
        source="test",
        mode="test",
    )

    # テストローダーの作成
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.get("batch_size", 128),
        shuffle=False,
        num_workers=args.get("num_workers", 4),
        pin_memory=True,
    )

    model._network.to(model._device)

    # --- [ADD] PRA 前の精度 ---
    # train
    cnn_tr_b, _ = _eval_on_loader(model, train_loader)
    logging.info(f"[PRA][Before][Train] CNN grouped: {cnn_tr_b.get('grouped', None)}")
    logging.info(f"[PRA][Before][Train] CNN retain_top1={cnn_tr_b.get('retain_top1', None)} "
                f"forget_top1={cnn_tr_b.get('forget_top1', None)} "
                f"forget_err={cnn_tr_b.get('forget_err', None)} "
                f"hmean={cnn_tr_b.get('hmean', None)}")

    # test
    cnn_te_b, _ = _eval_on_loader(model, test_loader)
    logging.info(f"[PRA][Before][Test]  CNN grouped: {cnn_te_b.get('grouped', None)}")
    logging.info(f"[PRA][Before][Test]  CNN retain_top1={cnn_te_b.get('retain_top1', None)} "
                f"forget_top1={cnn_te_b.get('forget_top1', None)} "
                f"forget_err={cnn_te_b.get('forget_err', None)} "
                f"hmean={cnn_te_b.get('hmean', None)}")

    
    
    #=== プロトタイプ再学習攻撃によって fc 層のパラメータを修正 ===#
    """
    model.forget_classes に含まれる class に対応する fc層のパラメータを修正
    """

    # プロトタイプ再学習用のデータローダーを作成
    data_memory = []
    target_memory = []
    for class_idx in model.forget_classes:

        # data と targets を取り出す
        data, targets, idx_dataset = data_manager.get_dataset(
            np.arange(class_idx, class_idx + 1),
            source="train",
            mode="test",
            ret_data=True,
        )

        data_memory.append(data)
        target_memory.append(targets)
    
    # 1クラスあたり何サンプル使うかを決定
    k_per_class = args.get("pra_num_samples_per_class", None)

    rng = np.random.default_rng(args["seed"])

    sampled_data = []
    sampled_targets = []

    for data, targets in zip(data_memory, target_memory):
        # data: (N, ...) / targets: (N,)
        if k_per_class is not None and k_per_class > 0 and len(targets) > k_per_class:
            idx = rng.choice(len(targets), size=k_per_class, replace=False)
            data = data[idx]
            targets = targets[idx]

        sampled_data.append(data)
        sampled_targets.append(targets)
    
    if len(sampled_data) > 0:
        data_memory_all = np.concatenate(sampled_data, axis=0)
        target_memory_all = np.concatenate(sampled_targets, axis=0)
    else:
        data_memory_all = np.array([])
        target_memory_all = np.array([])
    
    # データセットの作成
    pra_dataset = data_manager.get_dataset(
        [], source="train", mode="test", appendent=(data_memory_all, target_memory_all)
    )

    # データローダーの作成
    pra_loader = DataLoader(
        pra_dataset,
        batch_size=args.get("batch_size", 128),
        shuffle=False,                 # prototype平均を取るだけなら False が安定
        num_workers=args.get("num_workers", 4),
        pin_memory=True,
    )

    # プロトタイプ再学習によって fc 層を更新
    _apply_prototypical_relearning_attack(model, pra_loader, model.forget_classes, args)

    #=== プロトタイプ再学習攻撃を行なった後の精度評価 ===#
    model._network.to(model._device)  # 念のため（PRA内でもtoしてるが安全策） :contentReference[oaicite:9]{index=9}
    logging.info("[PRA] ===== After attack =====")
    
    # train
    cnn_tr_a, _ = _eval_on_loader(model, train_loader)
    logging.info(f"[PRA][After][Train] CNN grouped: {cnn_tr_a.get('grouped', None)}")
    logging.info(f"[PRA][After][Train] CNN retain_top1={cnn_tr_a.get('retain_top1', None)} "
                f"forget_top1={cnn_tr_a.get('forget_top1', None)} "
                f"forget_err={cnn_tr_a.get('forget_err', None)} "
                f"hmean={cnn_tr_a.get('hmean', None)}")

    # test
    cnn_te_a, _ = _eval_on_loader(model, test_loader)
    logging.info(f"[PRA][After][Test]  CNN grouped: {cnn_te_a.get('grouped', None)}")
    logging.info(f"[PRA][After][Test]  CNN retain_top1={cnn_te_a.get('retain_top1', None)} "
                f"forget_top1={cnn_te_a.get('forget_top1', None)} "
                f"forget_err={cnn_te_a.get('forget_err', None)} "
                f"hmean={cnn_te_a.get('hmean', None)}")

    


def _apply_prototypical_relearning_attack(model, pra_loader, forget_classes, args):
    """
    Prototypical Relearning Attack:
      - forget_classes の prototype を計算
      - fc の該当クラス行を prototype 方向へ更新
    """

    if forget_classes is None or len(forget_classes) == 0:
        logging.info("[PRA] forget_classes is empty. Skip.")
        return

    device = getattr(model, "_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    net = model._network
    # 念のため DataParallel 対応（eval 側で使うことは少ないけど安全策）
    if hasattr(net, "module"):
        net_ = net.module
    else:
        net_ = net

    net_.to(device)
    net_.eval()

    # --- PRA hyperparams ---
    # すでに loader 作成時に pra_num_samples_per_class で subsample 済みなら、
    # ここは「上限」として使うだけでOK（Noneなら全サンプル平均）
    max_per_class = args.get("pra_num_samples_per_class", None)
    metric = args.get("pra_metric", "cosine")          # "cosine" or "l2"
    normalize_proto = args.get("pra_normalize_proto", True)
    alpha = float(args.get("pra_alpha", 0.5))          # 0 -> no change, 1 -> replace with proto

    forget_set = set(int(c) for c in forget_classes)

    # --- prototype を running sum で計算（メモリ節約） ---
    feat_sums = {c: None for c in forget_set}
    counts = {c: 0 for c in forget_set}

    with torch.no_grad():
        for _, inputs, targets in pra_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 特徴量の取り出し
            feats = net_.extract_vector(inputs)  # [B, D]

            for feat, lbl in zip(feats, targets):
                c = int(lbl.item())
                if c not in counts:
                    continue
                if (max_per_class is not None) and (max_per_class > 0) and (counts[c] >= max_per_class):
                    continue

                if feat_sums[c] is None:
                    feat_sums[c] = feat.detach().clone()
                else:
                    feat_sums[c] += feat.detach()
                counts[c] += 1

            # 全クラスが上限に到達したら早期終了
            if (max_per_class is not None) and (max_per_class > 0):
                if all(counts[c] >= max_per_class for c in forget_set):
                    break

    proto_dict = {}
    for c in sorted(forget_set):
        if counts[c] == 0:
            logging.warning(f"[PRA] class {c}: no samples found in pra_loader. Skip.")
            continue
        proto = feat_sums[c] / float(counts[c])  # [D]
        if normalize_proto:
            proto = F.normalize(proto, p=2, dim=0)
        proto_dict[c] = proto

    logging.info(f"[PRA] computed prototypes for {len(proto_dict)}/{len(forget_set)} classes. "
                 f"counts={{{', '.join([str(k)+':'+str(counts[k]) for k in sorted(forget_set)])}}}")
    

    # --- fc 更新 ---
    fc = net_.fc
    if not hasattr(fc, "weight"):
        raise RuntimeError("[PRA] net.fc has no attribute 'weight'. This PRA impl assumes SimpleLinear/CosineLinear-like fc.")

    W = fc.weight.data  # [C, D]
    b = None
    if hasattr(fc, "bias") and (fc.bias is not None):
        b = fc.bias.data  # [C]

    for c, proto in proto_dict.items():
        if c < 0 or c >= W.size(0):
            logging.warning(f"[PRA] class {c}: out of range for fc (0..{W.size(0)-1}). Skip.")
            continue

        w_old = W[c].clone()

        if metric == "l2":
            # Spotter 実装に合わせた形：w=2p, b=-||p||^2（bias がある場合） 
            w_proto = (2.0 * proto).to(W.dtype)
            b_proto = (-proto.pow(2).sum()).to(W.dtype)
        else:
            # cosine
            w_proto = proto.to(W.dtype)
            b_proto = torch.tensor(0.0, device=device, dtype=W.dtype)

        W[c] = alpha * w_proto + (1.0 - alpha) * w_old

        if b is not None:
            b_old = b[c].clone()
            b[c] = alpha * b_proto + (1.0 - alpha) * b_old

    logging.info(f"[PRA] fc updated for forget classes (alpha={alpha}, metric={metric}, normalize_proto={normalize_proto}).")


def _eval_on_loader(model, loader):

    model.test_loader = loader
    return model.eval_task()

    



def _strip_module_prefix(state_dict):
    # DataParallel 保存の "module." を剥がす
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}

def _infer_num_convnets(state_dict):
    # "convnets.{i}." の i の最大値から本数を推定
    idxs = []
    for k in state_dict.keys():
        if k.startswith("convnets."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                idxs.append(int(parts[1]))
    return (max(idxs) + 1) if len(idxs) > 0 else None





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

