
import copy
import logging
import numpy as np
import os
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

from utils.grad_conflict import GradConflictLogger, select_named_params


EPSILON = 1e-8

# init_epoch = 200
# init_lr = 0.1
# init_milestones = [60, 120, 170]
# init_lr_decay = 0.1
# init_weight_decay = 0.0005


# epochs = 170
# lrate = 0.1
# milestones = [80, 120]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2


class iCaRLMU2(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)

        # Machine Unleaning パラメータ
        self.forget_list = args["forget_cls"]   # タスク毎の忘却予定リスト
        self.forget_classes = []

        self._retain_classes = 0

        # 学習ハイパラ（全タスク共通）
        self.batch_size = args["batch_size"]

        # ------------------------------------------------------------------
        # Replay sampling (NEW)
        #  - これまで: DataLoader の dataset に replay buffer を混ぜる
        #  - これから: 各 iteration で replay buffer から直接サンプルする
        #
        # ユーザ指定がなければ、従来の「混ぜ込み比率」を近似するため、
        #   replay batch size を (memory_len / (new_len + memory_len)) * batch_size
        # から自動決定する（task>0 のみ）。
        # ------------------------------------------------------------------
        self.replay_retain_batch_size = args.get("replay_retain_batch_size", None)
        self.replay_forget_batch_size = args.get("replay_forget_batch_size", 0)
        self.replay_with_replacement = bool(args.get("replay_with_replacement", True))

        # 内部状態（taskごとに更新）
        self._replay_dataset = None              # DummyDataset over (data_memory, targets_memory) w/ train transforms
        self._replay_targets_np = None           # np.ndarray, shape [M]
        self._replay_indices_retain = None       # np.ndarray of indices into replay dataset
        self._replay_indices_forget = None       # np.ndarray of indices into replay dataset
        self._batch_size_new = self.batch_size   # current-task DataLoader batch size (excluding replay)
        
        #　学習ハイパラ（初期タスク）
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_milestones = args["init_milestones"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]

        # 学習ハイパラ（後続タスク）
        self.epochs = args["epochs"]
        self.lr = args["lr"]
        self.milestones = args["milestones"]
        self.lr_decay = args["lr_decay"]
        self.weight_decay = args["weight_decay"]
        self.lambda_forg = args["lambda_forg"]
        self.T = args["T"]

        # 学習ハイパラ（損失重み）
        self.lambda_clf = args["lambda_clf"]
        self.lambda_kd = args["lambda_kd"]
        self.lambda_forg = args["lambda_forg"]

        # 忘却クラスのNME分類用
        self._forget_class_means = {}    # dict: class_id -> mean_vector (np.ndarray, shape [feature_dim])
        self._forget_class_targets = []  # list[int]

        # その他パラメータ
        self.num_workers = args["num_workers"]

        # ------------------------------------------------------------------
        # Replay sampling (変更点１＆２)
        #  - 旧: DataLoaderのdatasetにメモリをappendして混ぜる
        #  - 新: 毎iterationでリプレイバッファから直接サンプルを取得して学習
        #
        # total batch size は従来どおり self.batch_size を基準とし、
        #   new_batch + replay_retain_batch + replay_forget_batch = total
        # を満たすように構成する。
        # ------------------------------------------------------------------
        self.replay_retain_batch_size = args.get(
            "replay_retain_batch_size", args.get("replay_batch_size", None)
        )
        self.replay_forget_batch_size = args.get("replay_forget_batch_size", 0)

        # リプレイ用 dataset（train transform 適用済み）
        self._replay_dataset = None
        self._replay_targets = None  # np.ndarray
        self._replay_indices_all = None
        self._replay_indices_retain = None
        self._replay_indices_forget = None

        # どこかで1回だけ作る（__init__ か _train の冒頭が楽）
        if not hasattr(self, "_gc_logger"):
            if self.args.get("grad_conflict", False):
                save_dir = os.path.join(self.args["log_dir"], "grad_conflict")
                self._gc_logger = GradConflictLogger(save_dir, jsonl_name="icarl_mu.jsonl")
                self._gc_interval = int(self.args.get("grad_conflict_interval", 100))
                self._gc_include_fc = bool(self.args.get("grad_conflict_include_fc", False))
            else:
                self._gc_logger = None


    #---------------------------------------------------------------------------------------
    # タスク後の処理
    #---------------------------------------------------------------------------------------
    def after_task(self, log_dir=None):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

        #=== モデルの保存 ===#
        filename = f"{log_dir}phase"
        self.save_checkpoint(filename)
    
    def save_checkpoint(self, filename):
        # 既存BaseLearner互換：CPUに落として保存
        self._network.cpu()

        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
            "forget_classes": copy.deepcopy(getattr(self, "forget_classes", [])),
        }

        # 要らなければ後でコメントアウト
        if hasattr(self, "_forget_class_means"):
            save_dict["forget_class_means"] = copy.deepcopy(self._forget_class_means)
        if hasattr(self, "_forget_class_targets"):
            save_dict["forget_class_targets"] = copy.deepcopy(self._forget_class_targets)

        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    #---------------------------------------------------------------------------------------
    # モデルの訓練
    #---------------------------------------------------------------------------------------
    def incremental_train(self, data_manager):
        
        #=== data manager の登録 ===#
        self.data_manager = data_manager

        #=== 現在タスクの更新 ===#
        self._cur_task += 1

        #=== 現在タスクのクラスまでを含めた合計のクラス数を更新 ===#
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        #=== 忘却クラスの更新 ===#
        self.forget_classes += [cls for cls in self.forget_list[self._cur_task]]
        logging.info("forget classes on task{}: {}".format(self._cur_task, self.forget_classes))

        #=== モデルの出力層を更新 ===#
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info("model: {}".format(self._network_module_ptr))
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        #=== 訓練用データセットの作成 ===#
        # 変更点１:
        #   これまで: appendent=self._get_memory() で replay buffer を dataset に混ぜる
        #   これから: dataset は "現在タスクのデータのみ" とし、replay は iteration ごとに直接サンプル
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )

        # replay 用 dataset / index pool を準備（task>0 で memory があるときのみ）
        self._prepare_replay_sampling(data_manager)
        # replay batch size を自動決定（ユーザ指定がなければ）
        self._auto_configure_replay_batch_sizes(new_len=len(train_dataset))

        #=== 訓練用データローダーの作成 ===#
        # replay 分を除いた "現在タスクデータ" の batch size
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self._batch_size_new,
            shuffle=True,
            num_workers=self.num_workers,
        )

        #=== テスト用データセットの作成 ===#
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test"
        )

        #=== テスト用データローダーの作成 ===#
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        # データパラレルの設定
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        # 訓練を実行
        self._train(self.train_loader, self.test_loader)

        #=== 現在タスクまでの保持クラスを更新 ===#
        self._retain_classes = self._total_classes - len(self.forget_classes)

        # リプレイバッファの更新
        m = self._memory_size // self._retain_classes
        self.build_rehearsal_memory(data_manager, m)
        print("self._targets_memory.shape: ", self._targets_memory.shape)
        
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):

        #=== model をデバイスに配置 ===#
        self._network.to(self._device)

        #=== 知識蒸留用の教師モデル ===#
        if self._old_network is not None:
            self._old_network.to(self._device)

        #=== 1タスク目の学習 ===#
        if self._cur_task == 0:

            #=== Optimizer の設定 ===#
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.init_weight_decay,
            )

            #=== Scheduler の設定 ===#
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.init_milestones,
                gamma=self.init_lr_decay
            )

            #=== 学習の実行 ===#
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        
        #=== 2タスク目以降の学習 ===#
        else:

            #=== Optimizer の設定 ===#
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )  # 1e-5

            #=== Scheduler の設定 ===#
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.milestones,
                gamma=self.lr_decay
            )

            # 学習の実行
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    #=== 初期タスクの学習 ===#
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):

        #=== プログレスバーの設定 ===#
        prog_bar = tqdm(range(self.init_epoch))

        #=== 1エポックずつ学習 ===#
        for _, epoch in enumerate(prog_bar):

            #=== model を trainモード に変更
            self._network.train()

            #=== 記録用変数の初期化 ===#
            losses = 0.0
            correct = 0
            total = 0

            #=== 1エポックの学習 ===#
            for i, (_, inputs, targets) in enumerate(train_loader):
                
                # ----------------------------------------
                # ① 現在タスクのバッチを gpu に載せる
                # ----------------------------------------
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # ----------------------------------------
                # ② Forward 処理
                # ----------------------------------------
                logits = self._network(inputs)["logits"]

                # ----------------------------------------
                # ③ 損失計算
                # ----------------------------------------
                loss = F.cross_entropy(logits, targets)

                # ----------------------------------------
                # ④ 最適化処理
                # ----------------------------------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # ----------------------------------------
                # ⑤ 訓練精度の計算
                # ----------------------------------------
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
            logging.info(info)

        logging.info(info)

    #=== 2タスク目以降の学習 ===#
    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        #=== プログレスバーの設定 ===#
        prog_bar = tqdm(range(self.epochs))

        #=== 1エポックずつ学習 ===#
        for _, epoch in enumerate(prog_bar):

            #=== model を trainモード に変更
            self._network.train()

            #=== 記録用変数の初期化 ===#
            losses = 0.
            losses_clf = 0.
            losses_kd = 0.
            losses_forg = 0.
            correct = 0.
            total = 0.

            #=== 1エポックの学習 ===#
            for i, (_, inputs, targets) in enumerate(train_loader):

                # ----------------------------------------
                # ① 現在タスクのバッチを gpu に載せる
                # ----------------------------------------
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # ----------------------------------------
                # ② Replay buffer から iteration ごとに直接サンプル
                #    変更点１/２:
                #      - retain / forget を別々に指定して取り出せる
                # ----------------------------------------
                replay_inputs = []
                replay_targets = []

                # retain replay
                if self.replay_retain_batch_size and self.replay_retain_batch_size > 0:
                    rb = self.sample_replay_retain(self.replay_retain_batch_size)
                    if rb is not None:
                        replay_inputs.append(rb[0])
                        replay_targets.append(rb[1])

                # forget replay（必要なら）
                if self.replay_forget_batch_size and self.replay_forget_batch_size > 0:
                    fb = self.sample_replay_forget(self.replay_forget_batch_size)
                    if fb is not None:
                        replay_inputs.append(fb[0])
                        replay_targets.append(fb[1])

                if len(replay_inputs) > 0:
                    inputs_all = torch.cat([inputs] + replay_inputs, dim=0)
                    targets_all = torch.cat([targets] + replay_targets, dim=0)
                else:
                    inputs_all, targets_all = inputs, targets

                # ----------------------------------------
                # ③ 損失計算
                # ----------------------------------------
                logits, loss_clf, loss_kd, loss_forg = self.compute_loss(inputs_all, targets_all)
                loss = loss_clf * self.lambda_clf + loss_kd * self.lambda_kd +loss_forg * self.lambda_forg

                # ----------------------------------------
                # ③ 最適化処理
                # ----------------------------------------
                optimizer.zero_grad()

                # step は epoch / i から作る（例）
                global_step = epoch * len(train_loader) + i

                if self._gc_logger is not None and (global_step % self._gc_interval == 0):

                    # 「実際に optimizer が受け取る勾配」に合わせるなら、重み付き loss を渡すのが正解
                    loss_dict = {
                        "clf": loss_clf * self.lambda_clf,
                        "kd":  loss_kd  * self.lambda_kd,
                        "forg": loss_forg * self.lambda_forg,
                    }

                    # どの parameter で衝突を見るか（まずは backbone 推奨、fc除外）
                    def exclude_fc(name, p):
                        return (not self._gc_include_fc) and (".fc." in name or name.startswith("fc."))

                    named_params = select_named_params(self._network, exclude_fn=exclude_fc)
                    params = [p for _, p in named_params]

                    meta = {
                        "task": int(self._cur_task),
                        "epoch": int(epoch),
                        "iter": int(i),
                        "step": int(global_step),
                        "known": int(self._known_classes),
                        "total": int(self._total_classes),
                    }

                    # これが pairwise cosine / dot / norm を JSONL に吐く
                    self._gc_logger.log(loss_dict, params, meta=meta, retain_graph=True)
                    
                loss.backward()
                optimizer.step()

                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_kd += loss_kd.item()
                losses_forg += loss_forg.item()

                # ----------------------------------------
                # ⑤ 訓練精度の計算
                # ----------------------------------------
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets_all.expand_as(preds)).cpu().sum()
                total += len(targets_all)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Loss_forg {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1, self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
                    losses_forg / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Loss_forg {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
                    losses_forg / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)
        logging.info(info)

    #=== 損失の計算 ===#
    def compute_loss(self, inputs, targets):
        
        #=== model にサンプルを入力して logits を取得 ===#
        logits = self._network(inputs)["logits"]

        #=== forget / retain クラスを分割するためのマスク作成 ===#
        # バッチサイズ
        B = targets.shape[0]

        # forget_classes から 重複を削除（一応）
        forget_list = sorted(set(getattr(self, "forget_classes", [])))

        # 忘却クラスがない or 初期タスクならマスクなし
        if (self._cur_task == 0) or (len(forget_list) == 0):
            mask_forget = torch.zeros(B, device=targets.device, dtype=torch.bool)
        # 2タスク目以降ならマスクを生成
        else:
            # 忘却クラスを tensor 化
            forget_t = torch.as_tensor(forget_list, device=targets.device, dtype=targets.dtype)  # [F]
            
            # 忘却クラスを抜き出すためのマスクを作成
            mask_forget = torch.isin(targets, forget_t)  # [B]
        
        # 維持クラスを抜き出すためのマスクを作成
        mask_retain = ~mask_forget  # [B]

        #=== 交差エントロピー損失を計算 ===#
        loss_clf = torch.tensor(0., device=self._device)
        # loss_clf = F.cross_entropy(logits, targets)

        # 維持クラスがあれば損失計算
        if mask_retain.any():

            # retain クラスの logits と targets を取り出す
            retain_logits = logits[mask_retain]
            retain_targets = targets[mask_retain]

            loss_clf = F.cross_entropy(retain_logits, retain_targets)


        #=== 蒸留損失 loss_kd を計算 ===#
        loss_kd = torch.tensor(0., device=self._device)

         # 維持クラスがあれば損失計算
        if mask_retain.any():
            loss_kd = _KD_loss(
                logits[mask_retain, : self._known_classes],
                self._old_network(inputs)["logits"][mask_retain],
                self.T,
            )
        
        #=== 忘却損失の計算 ===#
        loss_forg = torch.tensor(0., device=self._device)

        # forget クラスの logits と targets を取り出す
        forget_logits = logits[mask_forget]
        forget_targets = targets[mask_forget]

        if mask_forget.any():
            forget_logits = logits[mask_forget]              # [Bf, C]
            log_p = F.log_softmax(forget_logits, dim=1)
            num_classes = log_p.size(1)
            uniform = torch.full_like(log_p, 1.0 / num_classes)  # target is prob (not log)

            loss_forg = F.kl_div(log_p, uniform, reduction="batchmean")


        return logits ,loss_clf, loss_kd, loss_forg

    #---------------------------------------------------------------------------------------
    # Replay sampling (変更点１＆２)
    #---------------------------------------------------------------------------------------
    def _prepare_replay_sampling(self, data_manager):
        """replay buffer を iteration ごとにサンプルするための準備。

        - 変更点１: DataLoader の dataset に replay buffer を混ぜない
        - 変更点２: replay buffer から retain / forget を別々に指定して取り出せるようにする
        """
        mem = self._get_memory()
        if mem is None:
            self._replay_dataset = None
            self._replay_targets_np = None
            self._replay_indices_retain = None
            self._replay_indices_forget = None
            return

        mem_data, mem_targets = mem
        if mem_targets is None or len(mem_targets) == 0:
            self._replay_dataset = None
            self._replay_targets_np = None
            self._replay_indices_retain = None
            self._replay_indices_forget = None
            return

        # train-time transform を適用するために DataManager から DummyDataset を作る
        self._replay_dataset = data_manager.get_dataset(
            [], source="train", mode="train", appendent=(mem_data, mem_targets)
        )
        self._replay_targets_np = np.asarray(mem_targets)

        # retain / forget の pool を作る
        if len(getattr(self, "forget_classes", [])) == 0:
            self._replay_indices_forget = np.asarray([], dtype=np.int64)
            self._replay_indices_retain = np.arange(len(self._replay_targets_np), dtype=np.int64)
        else:
            forget_set = np.asarray(sorted(set(self.forget_classes)), dtype=self._replay_targets_np.dtype)
            mask_forget = np.isin(self._replay_targets_np, forget_set)
            self._replay_indices_forget = np.where(mask_forget)[0].astype(np.int64)
            self._replay_indices_retain = np.where(~mask_forget)[0].astype(np.int64)
        

        # assert False

    def _auto_configure_replay_batch_sizes(self, new_len: int):
        """replay batch size を自動決定し、DataLoader 側の batch size を調整する。

        total batch size は self.batch_size（従来と同じ）を基準として、
          new_batch + replay_retain_batch + replay_forget_batch = total
        となるようにする。

        ユーザが `replay_retain_batch_size` を指定していれば、それを優先する。
        """
        # mem_len = int(self.exemplar_size)
        mem_len = int(self._targets_memory.shape[0])

        # task0 または memory がない場合は replay なし
        if self._cur_task == 0 or mem_len == 0:
            # self.replay_retain_batch_size = 0
            # self.replay_forget_batch_size = 0
            self._batch_size_new = int(self.batch_size)
            return

        # retain replay size（未指定なら従来の混ぜ込み比率を近似）
        if self.replay_retain_batch_size is None:
            total_len = max(1, int(new_len) + mem_len)
            exp_replay = int(round(float(self.batch_size) * mem_len / total_len))
            if exp_replay <= 0:
                exp_replay = 1
            self.replay_retain_batch_size = exp_replay

        # 型を整える
        self.replay_retain_batch_size = int(self.replay_retain_batch_size)
        self.replay_forget_batch_size = int(self.replay_forget_batch_size)
        # print("self.replay_retain_batch_size: ", self.replay_retain_batch_size)
        # print("self.replay_forget_batch_size: ", self.replay_forget_batch_size)
        # assert False

        # batch size が破綻しないように clamp
        if self.replay_forget_batch_size < 0:
            self.replay_forget_batch_size = 0
        if self.replay_retain_batch_size < 0:
            self.replay_retain_batch_size = 0

        if self.replay_retain_batch_size + self.replay_forget_batch_size >= self.batch_size:
            # new_batch を最低 1 確保
            max_total_replay = self.batch_size - 1
            # forget を優先して retain を削る（不要なら逆でもOK）
            self.replay_forget_batch_size = min(self.replay_forget_batch_size, max_total_replay)
            self.replay_retain_batch_size = max(0, max_total_replay - self.replay_forget_batch_size)

        self._batch_size_new = int(self.batch_size) - self.replay_retain_batch_size - self.replay_forget_batch_size
        self._batch_size_new = max(1, self._batch_size_new)

        logging.info(
            f"Replay config (task={self._cur_task}): new_batch={self._batch_size_new}, "
            f"replay_retain_batch={self.replay_retain_batch_size}, replay_forget_batch={self.replay_forget_batch_size}"
        )

    def _sample_replay_from_pool(self, pool_indices: np.ndarray, n: int):
        """pool_indices から n 個サンプルして (inputs, targets) を返す。"""
        if self._replay_dataset is None or self._replay_targets_np is None:
            return None
        if n is None or int(n) <= 0:
            return None
        if pool_indices is None or len(pool_indices) == 0:
            return None

        n = int(n)
        pool_indices = np.asarray(pool_indices, dtype=np.int64)
        replace = self.replay_with_replacement or (len(pool_indices) < n)
        chosen = np.random.choice(pool_indices, size=n, replace=replace)

        imgs = []
        labs = []
        for idx in chosen:
            item = self._replay_dataset[int(idx)]
            # DummyDataset(aug=1) : (idx, image, label)
            # aug>1 の場合: (idx, img1, img2, ..., label) になるので最初の view を取る
            label = item[-1]
            img = item[1]
            imgs.append(img)
            labs.append(label)

        inputs = torch.stack(imgs, dim=0).to(self._device, non_blocking=True)
        targets = torch.as_tensor(labs, device=self._device)
        return inputs, targets

    def sample_replay_retain(self, n: int, retain_classes: list = None):
        """replay buffer から retain クラスのみをサンプルする。

        Args:
            n: サンプル数
            retain_classes: 取り出したい retain class id の list（None なら retain pool 全体）
        """
        if self._replay_indices_retain is None:
            return None
        pool = self._replay_indices_retain
        if retain_classes is not None and self._replay_targets_np is not None:
            retain_classes = np.asarray(retain_classes, dtype=self._replay_targets_np.dtype)
            mask = np.isin(self._replay_targets_np[pool], retain_classes)
            pool = pool[mask]
        return self._sample_replay_from_pool(pool, n)

    def sample_replay_forget(self, n: int, forget_classes: list = None):
        """replay buffer から forget クラスのみをサンプルする。

        Args:
            n: サンプル数
            forget_classes: 取り出したい forget class id の list（None なら forget pool 全体）
        """
        if self._replay_indices_forget is None:
            return None
        pool = self._replay_indices_forget
        if forget_classes is not None and self._replay_targets_np is not None:
            forget_classes = np.asarray(forget_classes, dtype=self._replay_targets_np.dtype)
            mask = np.isin(self._replay_targets_np[pool], forget_classes)
            pool = pool[mask]
        return self._sample_replay_from_pool(pool, n)

    #---------------------------------------------------------------------------------------
    # リプレイバッファの構築
    #---------------------------------------------------------------------------------------
    def build_rehearsal_memory(self, data_manager, per_class):
        
        #=== クラス毎に固定数 per_class を保存する場合 ===#
        if self._fixed_memory:
            assert False
            self._construct_exemplar_unified(data_manager, per_class)
        #=== リプレイバッファのサイズを固定し，self_retain_classes の数で均等に分割する場合 ===#
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))

        #=== 学習済み保持クラスのリストを作成 ===#
        forget_seen = {c for c in self.forget_classes if 0 <= c < self._total_classes}
        retain_classes = [c for c in range(self._total_classes) if c not in forget_seen]
        
        #=== 以前のリプレイバッファの内容をコピー ===#
        dummy_data = copy.deepcopy(self._data_memory)
        dummy_targets = copy.deepcopy(self._targets_memory)

        #=== リプレイバッファの内容を空に初期化 ===#
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory = np.array([])
        self._targets_memory = np.array([])

        for class_idx in range(self._known_classes):

            mask = np.where(dummy_targets == class_idx)[0]
            if len(mask) == 0:
                continue

            # forget クラスはスキップ
            if class_idx in forget_seen:

                # 忘却クラスの サンプル を “全部” 使って mean を計算
                dd = dummy_data[mask]
                dt = dummy_targets[mask]

                # mean をまだ保存していない場合だけ（再計算したいならこのifを外す）
                if class_idx not in self._forget_class_means:
                    idx_dataset = data_manager.get_dataset(
                        [], source="train", mode="test", appendent=(dd, dt)
                    )
                    idx_loader = DataLoader(
                        idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
                    )
                    vectors, _ = self._extract_vectors(idx_loader)
                    vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                    mean = np.mean(vectors, axis=0)
                    n = np.linalg.norm(mean)
                    if n > 0:
                        mean = mean / n

                    self._forget_class_means[class_idx] = mean
                    if class_idx not in self._forget_class_targets:
                        self._forget_class_targets.append(class_idx)

                continue  # ★ forget class はバッファに残さない

            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):

        logging.info("Constructing exemplars...({} per classes)".format(m))
        
        forget_seen = {c for c in self.forget_classes if 0 <= c < self._total_classes}

        for class_idx in range(self._known_classes, self._total_classes):

            if class_idx in forget_seen:
                continue  # ★ forget class はバッファに追加しない
            if m == 0:
                continue

            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean



    #---------------------------------------------------------------------------------------
    # 評価関連の処理
    #---------------------------------------------------------------------------------------
    def accuracy(self, y_pred, y_true, nb_old, init_cls=50, increment=10):
        assert len(y_pred) == len(y_true), "Data length error."
        all_acc = {}
        all_acc["total"] = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

        max_cls = int(np.max(y_true))

        # --- Group 0: [0 .. init_cls-1] ---
        start, end = 0, min(init_cls - 1, max_cls)
        idxes = np.where(np.logical_and(y_true >= start, y_true <= end))[0]
        label = "{}-{}".format(str(start).rjust(2, "0"), str(end).rjust(2, "0"))
        all_acc[label] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

        # --- Next groups: [init_cls ..] with step=increment ---
        for start in range(init_cls, max_cls + 1, increment):
            end = min(start + increment - 1, max_cls)
            idxes = np.where(np.logical_and(y_true >= start, y_true <= end))[0]
            label = "{}-{}".format(str(start).rjust(2, "0"), str(end).rjust(2, "0"))
            all_acc[label] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

        # Old/New accuracy は元のまま
        idxes = np.where(y_true < nb_old)[0]
        all_acc["old"] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

        idxes = np.where(y_true >= nb_old)[0]
        all_acc["new"] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

        return all_acc

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = self.accuracy(y_pred.T[0],
                                y_true,
                                self._known_classes,
                                init_cls=self.args["init_cls"],
                                increment=self.args["increment"])
        
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def eval_task(self):
        
        task_id = getattr(self, "_cur_task", -1)

        def _fmt(x):
            return "N/A" if x is None else f"{float(x):.2f}"

        # -------------------------
        # CNN 評価
        # -------------------------
        #=== model の forward 処理 ===#
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)  # 既存どおり（全クラスの top1/top5/grouped）

        y_true = np.asarray(y_true)
        top1 = y_pred[:, 0]

        #=== 忘却クラス / 保持クラスを分割するためのマスクを作成 ===#
        forget_set = {c for c in getattr(self, "forget_classes", []) if 0 <= c < self._total_classes}
        if len(forget_set) > 0:
            mask_forget = np.isin(y_true, list(forget_set))
        else:
            mask_forget = np.zeros_like(y_true, dtype=bool)
        mask_retain = ~mask_forget

        def _top1_acc(mask):
            if not mask.any():
                return None
            return np.around((top1[mask] == y_true[mask]).sum() * 100.0 / mask.sum(), decimals=2)

        #=== 保持クラスのみで精度を計算 ===#
        retain_top1 = _top1_acc(mask_retain)
        
        #=== 忘却クラスのみで精度を計算 ===#
        forget_top1 = _top1_acc(mask_forget)

        cnn_accy["retain_top1"] = retain_top1
        cnn_accy["forget_top1"] = forget_top1
        cnn_accy["num_retain_samples"] = int(mask_retain.sum())
        cnn_accy["num_forget_samples"] = int(mask_forget.sum())

        if (retain_top1 is not None) and (forget_top1 is not None):
            forget_err = 100.0 - forget_top1
            denom = retain_top1 + forget_err
            hmean = 0.0 if denom <= 0 else (2.0 * retain_top1 * forget_err / denom)
        else:
            forget_err, hmean = None, None

        cnn_accy["forget_err"] = forget_err
        cnn_accy["hmean"] = hmean

        logging.info(
            f"[Eval][Task {task_id}] CNN | "
            f"retain_top1={_fmt(cnn_accy['retain_top1'])} "
            f"forget_top1={_fmt(cnn_accy['forget_top1'])} "
            f"forget_err={_fmt(cnn_accy['forget_err'])} "
            f"hmean={_fmt(cnn_accy['hmean'])} | "
            f"retain_n={cnn_accy['num_retain_samples']} "
            f"forget_n={cnn_accy['num_forget_samples']}"
        )


        # -------------------------
        # NME 評価
        # -------------------------
        if hasattr(self, "_class_means"):

            means_for_eval = self._class_means.copy()
            for c, v in getattr(self, "_forget_class_means", {}).items():
                if 0 <= c < self._total_classes:
                    means_for_eval[c] = v


            y_pred_nme, y_true_nme = self._eval_nme(self.test_loader, means_for_eval)
            # y_pred_nme, y_true_nme = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred_nme, y_true_nme)

            y_true_nme = np.asarray(y_true_nme)
            top1_nme = y_pred_nme[:, 0]
            if len(forget_set) > 0:
                mask_forget_nme = np.isin(y_true_nme, list(forget_set))
            else:
                mask_forget_nme = np.zeros_like(y_true_nme, dtype=bool)
            mask_retain_nme = ~mask_forget_nme

            def _top1_acc_nme(mask):
                if not mask.any():
                    return None
                return np.around((top1_nme[mask] == y_true_nme[mask]).sum() * 100.0 / mask.sum(), decimals=2)

            retain_top1_nme = _top1_acc_nme(mask_retain_nme)
            forget_top1_nme = _top1_acc_nme(mask_forget_nme)

            nme_accy["retain_top1"] = retain_top1_nme
            nme_accy["forget_top1"] = forget_top1_nme
            nme_accy["num_retain_samples"] = int(mask_retain_nme.sum())
            nme_accy["num_forget_samples"] = int(mask_forget_nme.sum())

            if (retain_top1_nme is not None) and (forget_top1_nme is not None):
                forget_err_nme = 100.0 - forget_top1_nme
                denom = retain_top1_nme + forget_err_nme
                hmean_nme = 0.0 if denom <= 0 else (2.0 * retain_top1_nme * forget_err_nme / denom)
            else:
                forget_err_nme, hmean_nme = None, None

            nme_accy["forget_err"] = forget_err_nme
            nme_accy["hmean"] = hmean_nme
        else:
            nme_accy = None

        if nme_accy is not None:
            logging.info(
                f"[Eval][Task {task_id}] NME | "
                f"retain_top1={_fmt(nme_accy['retain_top1'])} "
                f"forget_top1={_fmt(nme_accy['forget_top1'])} "
                f"forget_err={_fmt(nme_accy['forget_err'])} "
                f"hmean={_fmt(nme_accy['hmean'])} | "
                f"retain_n={nme_accy['num_retain_samples']} "
                f"forget_n={nme_accy['num_forget_samples']}"
            )


        return cnn_accy, nme_accy



def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
