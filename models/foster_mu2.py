import copy
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from models.base import BaseLearner
from utils.inc_net import FOSTERNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

# Please refer to https://github.com/G-U-N/ECCV22-FOSTER for the full source code to reproduce foster.

EPSILON = 1e-8


class FOSTERMU2(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self._network = FOSTERNet(args, False)

        # Machine Unleaning パラメータ
        self.forget_list = args["forget_cls"]   # タスク毎の忘却予定リスト
        self.forget_classes = []

        self._retain_classes = 0

        # 学習ハイパラ（損失重み）
        self.lambda_clf = args["lambda_clf"]
        self.lambda_fe = args["lambda_fe"]
        self.lambda_kd = args["lambda_kd"]
        self.lambda_forg = args["lambda_forg"]

        # 忘却クラスのNME分類用
        self._forget_class_means = {}    # dict: class_id -> mean_vector (np.ndarray, shape [feature_dim])
        self._forget_class_targets = []  # list[int]

        # その他パラメータ
        self.num_workers = args["num_workers"]

        
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.init_epochs = args["init_epochs"]
        self.init_weight_decay = args["init_weight_decay"]

        self.lr = args["lr"]
        self.epochs = args["boosting_epochs"]
        
        self._snet = None
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.per_cls_weights = None
        self.is_teacher_wa = args["is_teacher_wa"]
        self.is_student_wa = args["is_student_wa"]
        self.lambda_okd = args["lambda_okd"]
        self.wa_value = args["wa_value"]
        self.oofc = args["oofc"].lower()

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
        # if self._cur_task > 1:
        #     self._network = self._snet
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        #=== 現在タスクのクラスまでを含めた合計のクラス数を更新 ===#
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        #=== 忘却クラスの更新 ===#
        self.forget_classes += [cls for cls in self.forget_list[self._cur_task]]
        logging.info("forget classes on task{}: {}".format(self._cur_task, self.forget_classes))

        #=== モデルの出力層を更新 ===#
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        #=== 並列モデルはパラメータを固定 ===#
        if self._cur_task > 0:
            for p in self._network.convnets[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        #=== 訓練用データセットの作成 ===#
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )

        #=== 訓練用データローダーの作成 ===#
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
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

        # 生徒ネットを更新
        if not self._cur_task == 0:
            self._network = self._snet

        #=== 現在タスクまでの保持クラスを更新 ===#
        self._retain_classes = self._total_classes - len(self.forget_classes)

        # リプレイバッファの更新
        if self._fixed_memory:
            m = self._memory_per_class  # または self.samples_old_class
        else:
            m = self._memory_size // self._retain_classes
        self.build_rehearsal_memory(data_manager, m)
        print("self._targets_memory.shape: ", self._targets_memory.shape)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network_module_ptr.train()
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            self._network_module_ptr.convnets[0].eval()

    def _train(self, train_loader, test_loader):

        #=== model をデバイスに配置 ===#
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module

        #=== 1タスク目の学習 ===#
        if self._cur_task == 0:

            #=== Optimizer の設定 ===#
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.init_weight_decay,
            )

            #=== Scheduler の設定 ===#
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.args["init_epochs"]
            )

            #=== 学習の実行 ===#
            self._init_train(
                train_loader,
                test_loader,
                optimizer,
                scheduler
            )

        #=== 2タスク目以降の学習 ===#
        else:
            
            #=== 各クラスのサンプル数に応じて重みを作成 ===#
            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]

            effective_num = 1.0 - np.power(self.beta1, cls_num_list)
            per_cls_weights = (1.0 - self.beta1) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )

            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)

            #=== Optimizer の設定 ===#
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lr"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )

            #=== Scheduler の設定 ===#
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.args["boosting_epochs"]
            )

            #=== マスクの作成 ===#
            if self.oofc == "az":
                for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                    if i == 0:
                        p.data[
                            self._known_classes :, : self._network_module_ptr.out_dim
                        ] = torch.tensor(0.0)
            elif self.oofc != "ft":
                assert 0, "not implemented"

            # 学習の実行
            self._feature_boosting(train_loader, test_loader, optimizer, scheduler)
            if self.is_teacher_wa:
                self._network_module_ptr.weight_align(
                    self._known_classes,
                    self._total_classes - self._known_classes,
                    self.wa_value,
                )
            else:
                logging.info("do not weight align teacher!")
            
            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]
            effective_num = 1.0 - np.power(self.beta2, cls_num_list)
            per_cls_weights = (1.0 - self.beta2) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)
            self._feature_compression(train_loader, test_loader)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):

        #=== プログレスバーの設定 ===#
        prog_bar = tqdm(range(self.init_epochs))

        #=== 1エポックずつ学習 ===#
        for _, epoch in enumerate(prog_bar):

            #=== model を trainモード に変更
            self.train()

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
                    self.init_epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.init_epochs,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
            logging.info(info)

        logging.info(info)

    def _feature_boosting(self, train_loader, test_loader, optimizer, scheduler):
        
        #=== プログレスバーの設定 ===#
        prog_bar = tqdm(range(self.args["boosting_epochs"]))

        #=== 1エポックずつ学習 ===#
        for _, epoch in enumerate(prog_bar):

            #=== model を trainモード に変更
            self.train()

            losses = 0.0
            losses_clf = 0.0
            losses_fe = 0.0
            losses_kd = 0.0
            losses_forg = 0.0

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
                # ② 損失計算
                # ----------------------------------------
                logits, loss_clf, loss_fe, loss_kd, loss_forg = self.compute_loss(inputs, targets)
                loss = loss_clf * self.lambda_clf + loss_fe * self.lambda_fe + loss_kd +loss_forg * self.lambda_forg

                # ----------------------------------------
                # ③ 最適化処理
                # ----------------------------------------
                optimizer.zero_grad()
                loss.backward()
                if self.oofc == "az":
                    for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                        if i == 0:
                            p.grad.data[
                                self._known_classes :,
                                : self._network_module_ptr.out_dim,
                            ] = torch.tensor(0.0)
                elif self.oofc != "ft":
                    assert 0, "not implemented"
                optimizer.step()

                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_fe += loss_fe.item()
                losses_kd += loss_kd.item()
                losses_forg += loss_forg.item()

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
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Loss_forg {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1, self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    losses_forg / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Loss_forg {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    losses_forg / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)
        logging.info(info)

    def _feature_compression(self, train_loader, test_loader):

        #=== 初期化モデルを用意 ===#
        self._snet = FOSTERNet(self.args, False)
        self._snet.update_fc(self._total_classes)

        #=== データパラレルの用意 ===#
        if len(self._multiple_gpus) > 1:
            self._snet = nn.DataParallel(self._snet, self._multiple_gpus)
        if hasattr(self._snet, "module"):
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet

        #=== モデルをデバイス上に配置 ===#
        self._snet.to(self._device)

        self._snet_module_ptr.convnets[0].load_state_dict(
            self._network_module_ptr.convnets[0].state_dict()
        )
        self._snet_module_ptr.copy_fc(self._network_module_ptr.oldfc)

        #=== Optimizer の作成 ===#
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._snet.parameters()),
            lr=self.args["lr"],
            momentum=0.9,
        )

        #=== Scheduler の作成 ===#
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.args["compression_epochs"]
        )

        #=== 教師モデルを評価モードに変更 ===#
        self._network.eval()

        #=== プログレスバーの設定 ===#
        prog_bar = tqdm(range(self.args["compression_epochs"]))

        #=== 1エポックずつ学習 ===#
        for _, epoch in enumerate(prog_bar):

            #=== 生徒モデルを訓練モードに変更 ===#
            self._snet.train()

            #=== 記録用変数の初期化 ===#
            losses = 0.0
            losses_kd = 0.0
            losses_forg = 0.0
            correct, total = 0, 0

            #=== 1エポックの学習 ===#
            for i, (_, inputs, targets) in enumerate(train_loader):

                # ----------------------------------------
                # ① 現在タスクのバッチを gpu に載せる
                # ----------------------------------------
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # ----------------------------------------
                # ② 損失計算
                # ----------------------------------------
                s_out = self._snet(inputs)
                dark_logits = s_out["logits"]
                dark_feats = s_out["features"]

                # --- teacher forward ---
                with torch.no_grad():
                    t_out = self._network(inputs)
                    teacher_logits = t_out["logits"]

                # --- forget/retain mask ---
                mask_forget = self._make_forget_mask(targets)
                mask_retain = ~mask_forget

                # --- 蒸留損失 ---
                loss_dark = torch.tensor(0.0, device=self._device)
                if mask_retain.any():
                    loss_dark = self.BKD(
                        dark_logits[mask_retain],
                        teacher_logits[mask_retain],
                        self.args["T"],
                    )

                # --- 忘却損失 ---
                loss_fcos = torch.tensor(0.0, device=self._device)
                # if mask_forget.any():
                #     loss_fcos = self._forget_cosine_min_loss(
                #         dark_feats[mask_forget],
                #         targets[mask_forget],
                #     )

                loss = loss_dark + self.lambda_forg * loss_fcos

                # ----------------------------------------
                # ③ 最適化処理
                # ----------------------------------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

                # ----------------------------------------
                # ⑤ 訓練精度の計算
                # ----------------------------------------
                _, preds = torch.max(dark_logits[: targets.shape[0]], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f}, Loss_kd {:.3f}, Loss_forg {:.3f},  Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    losses_kd / len(train_loader),
                    losses_forg / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f}, Loss_kd {:.3f}, Loss_forg {:.3f},  Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    losses_kd / len(train_loader),
                    losses_forg / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)
        if len(self._multiple_gpus) > 1:
            self._snet = self._snet.module
        if self.is_student_wa:
            self._snet.weight_align(
                self._known_classes,
                self._total_classes - self._known_classes,
                self.wa_value,
            )
        else:
            logging.info("do not weight align student!")


        self._snet.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = self._snet(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cnn_accy = self._evaluate(y_pred, y_true)
        logging.info("darknet eval: ")
        logging.info("CNN top1 curve: {}".format(cnn_accy["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_accy["top5"]))


    def compute_loss(self, inputs, targets):

        #=== model にサンプルを入力して outputs を取得 ===#
        outputs = self._network(inputs)

        #=== logits を分解 ===#
        logits, fe_logits, old_logits = (
            outputs["logits"],
            outputs["fe_logits"],
            outputs["old_logits"].detach(),
        )
        features = outputs["features"]

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

        # 維持クラスがあれば損失計算
        if mask_retain.any():

            # retain クラスの logits と targets を取り出す
            retain_logits = logits[mask_retain]
            retain_targets = targets[mask_retain]

            loss_clf = F.cross_entropy(retain_logits / self.per_cls_weights, retain_targets)

        #=== FE 損失を計算 ===#
        loss_fe = torch.tensor(0., device=self._device)

        # 維持クラスがあれば損失計算
        if mask_retain.any():

            # retain クラスの logits と targets を取り出す
            retain_fe_logits = fe_logits[mask_retain]
            retain_targets = targets[mask_retain]

            loss_fe = F.cross_entropy(retain_fe_logits, retain_targets)

        #=== 蒸留損失を計算 ===#
        loss_kd = torch.tensor(0., device=self._device)

        # 維持クラスがあれば損失計算
        if mask_retain.any():

            retain_logits_old = retain_logits[:, : self._known_classes]     # [Br, known]
            retain_old_logits = old_logits[mask_retain]                    # [Br, known]
            loss_kd = self.lambda_okd * _KD_loss(
                retain_logits_old, retain_old_logits, self.args["T"]
            )


        #=== 忘却損失を計算 ===#
        loss_forg = torch.tensor(0., device=self._device)
        loss_forg2 = torch.tensor(0., device=self._device)

        if mask_forget.any():
            loss_forg = self._forget_cosine_min_loss(
                features[mask_forget],
                targets[mask_forget],
            )
        
        if mask_forget.any():
            forget_logits = logits[mask_forget]            # [Bf, C]
            log_p = F.log_softmax(forget_logits, dim=1)
            num_classes = log_p.size(1)
            uniform = torch.full_like(log_p, 1.0 / num_classes)
            loss_forg2 = F.kl_div(log_p, uniform, reduction="batchmean")

        loss_forg += loss_forg2 * 0.1

        return logits, loss_clf, loss_fe, loss_kd, loss_forg

    def _make_forget_mask(self, targets: torch.Tensor) -> torch.Tensor:
        B = targets.shape[0]
        forget_list = sorted(set(getattr(self, "forget_classes", [])))

        if (self._cur_task == 0) or (len(forget_list) == 0):
            return torch.zeros(B, device=targets.device, dtype=torch.bool)

        forget_t = torch.as_tensor(forget_list, device=targets.device, dtype=targets.dtype)
        return torch.isin(targets, forget_t)


    def _forget_cosine_min_loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        忘却クラス c に属するサンプルに対して、
        同一クラス内の cosine similarity を最小化する loss。
        features: [Bf, D], targets: [Bf]
        """
        if features.ndim != 2:
            features = features.view(features.size(0), -1)

        feats = F.normalize(features, dim=1)
        loss = torch.tensor(0.0, device=features.device)
        n_cls = 0

        for c in targets.unique():
            idx = (targets == c)
            n = int(idx.sum().item())
            if n < 2:
                continue

            f = feats[idx]               # [n, D]
            sim = f @ f.t()              # [n, n]
            sim_sum = sim.sum() - sim.diag().sum()
            loss_c = sim_sum / (n * (n - 1))   # 平均ペア類似度
            loss = loss + loss_c
            n_cls += 1

        if n_cls > 0:
            loss = loss / n_cls
        return loss

    def BKD(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        soft = soft * self.per_cls_weights
        soft = soft / soft.sum(1)[:, None]
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
    

    @property
    def samples_old_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._known_classes

    def samples_new_class(self, index):
        if self.args["dataset"] == "cifar100":
            return 500
        else:
            return self.data_manager.getlen(index)

    #---------------------------------------------------------------------------------------
    # リプレイバッファの構築
    #---------------------------------------------------------------------------------------
    def build_rehearsal_memory(self, data_manager, per_class):
        
        #=== クラス毎に固定数 per_class を保存する場合 ===#
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        #=== リプレイバッファのサイズを固定し，self_retain_classes の数で均等に分割する場合 ===#
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )

        # --- forget class set ---
        forget_seen = {c for c in getattr(self, "forget_classes", [])
                    if 0 <= c < self._total_classes}

        # --- helper: mean calc from (dd, dt) ---
        def _calc_mean_from_appendent(dd, dt):
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
            return mean

        # =========================================================
        # (A) まずメモリ内の forget exemplars を退避→削除
        # =========================================================
        if len(forget_seen) > 0 and len(getattr(self, "_targets_memory", [])) > 0:
            for c in sorted(forget_seen):
                idx = np.where(self._targets_memory == c)[0]
                if len(idx) == 0:
                    continue

                # 忘却クラスの mean を「削除前」に計算して保存（方針1）
                if c not in self._forget_class_means:
                    dd = self._data_memory[idx]
                    dt = self._targets_memory[idx]
                    self._forget_class_means[c] = _calc_mean_from_appendent(dd, dt)
                    if c not in self._forget_class_targets:
                        self._forget_class_targets.append(c)

            # forget exemplars をメモリから削除
            keep_mask = ~np.isin(self._targets_memory, list(forget_seen))
            self._data_memory = self._data_memory[keep_mask]
            self._targets_memory = self._targets_memory[keep_mask]

        # class means（retain 側だけ更新。forget は _forget_class_means で管理）
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # =========================================================
        # (B) old classes の mean を「残った exemplars」で更新（forget はスキップ）
        # =========================================================
        for class_idx in range(self._known_classes):

            if class_idx in forget_seen:
                continue  # ★ forget class はメモリにいない前提

            mask = np.where(self._targets_memory == class_idx)[0]
            if len(mask) == 0:
                continue

            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]
            mean = _calc_mean_from_appendent(class_data, class_targets)
            _class_means[class_idx, :] = mean

        # =========================================================
        # (C) new classes: exemplar 構築（forget は作らない）
        # =========================================================
        for class_idx in range(self._known_classes, self._total_classes):

            # 忘却対象の新クラスなら、exemplar は作らず mean だけ保存してスキップ
            if class_idx in forget_seen:
                if class_idx not in self._forget_class_means:
                    data, targets, class_dset = data_manager.get_dataset(
                        np.arange(class_idx, class_idx + 1),
                        source="train",
                        mode="test",
                        ret_data=True,
                    )
                    mean = _calc_mean_from_appendent(data, targets)
                    self._forget_class_means[class_idx] = mean
                    if class_idx not in self._forget_class_targets:
                        self._forget_class_targets.append(class_idx)
                continue

            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select (herding)
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(exemplar_vectors, axis=0) if len(exemplar_vectors) > 0 else 0
                mu_p = (vectors + S) / k
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(np.array(data[i]))
                exemplar_vectors.append(np.array(vectors[i]))

                vectors = np.delete(vectors, i, axis=0)
                data = np.delete(data, i, axis=0)

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
            mean = _calc_mean_from_appendent(selected_exemplars, exemplar_targets)
            _class_means[class_idx, :] = mean

        # self._class_means = _class_means

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))

        #=== 学習済み保持クラスのリストを作成 ===#
        forget_seen = {c for c in self.forget_classes if 0 <= c < self._total_classes}
        retain_classes = [c for c in range(self._total_classes) if c not in forget_seen]
        
        #=== 以前のリプレイバッファの内容をコピー ===#
        dummy_data = copy.deepcopy(self._data_memory)
        dummy_targets = copy.deepcopy(self._targets_memory)

        #=== リプレイバッファの内容を空に初期化 ===#
        # self._class_means = np.zeros((self._total_classes, self.feature_dim))
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

            # self._class_means[class_idx, :] = mean

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

            # self._class_means[class_idx, :] = mean



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
        # NME 評価（DERでは無効化）
        # -------------------------
        nme_accy = None
        # logging.info(f"[Eval][Task {task_id}] NME disabled for DER (return None).")


        return cnn_accy, nme_accy


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]



