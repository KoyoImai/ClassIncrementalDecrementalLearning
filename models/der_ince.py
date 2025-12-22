# Please note that the current implementation of DER only contains the dynamic expansion process, since masking and pruning are not implemented by the source repo.
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
from utils.inc_net import INCEDERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPSILON = 1e-8

# init_epoch = 200
# init_lr = 0.1
# init_milestones = [60, 120, 170]
# init_lr_decay = 0.1
# init_weight_decay = 0.0005


# epochs = 170
# lrate = 0.1
# milestones = [80, 120, 150]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
# num_workers = 8
# T = 2


class DERINCE(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = INCEDERNet(args, False)

        # Machine Unleaning パラメータ
        self.forget_list = args["forget_cls"]   # タスク毎の忘却予定リスト
        self.forget_classes = []

        self._retain_classes = 0

        # 学習ハイパラ（全タスク共通）
        self.batch_size = args["batch_size"]
        
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
        self.lambda_aux = args["lambda_aux"]
        self.lambda_forg = args["lambda_forg"]

        # 忘却クラスのNME分類用
        self._forget_class_means = {}    # dict: class_id -> mean_vector (np.ndarray, shape [feature_dim])
        self._forget_class_targets = []  # list[int]

        # その他パラメータ
        self.num_workers = args["num_workers"]

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
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        #=== 並列モデルはパラメータを固定 ===#
        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
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
            num_workers=self.num_workers
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

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1 :
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()
    
    def _train(self, train_loader, test_loader):

        #=== model をデバイスに配置 ===#
        self._network.to(self._device)

        #=== 1タスク目の学習 ===#
        if self._cur_task == 0:

            #=== Optimizer の設定 ===#
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.lr,
                weight_decay=self.init_weight_decay,
            )

            #=== Scheduler の設定 ===#
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.init_milestones,
                gamma=self.init_lr_decay
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

            #=== Optimizer の設定 ===#
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )

            #=== Scheduler の設定 ===#
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.milestones,
                gamma=self.lr_decay
            )

            # 学習の実行
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            
            # 
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(
                    self._total_classes - self._known_classes
                )
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        
        #=== プログレスバーの設定 ===#
        prog_bar = tqdm(range(self.init_epoch))

        #=== 1エポックずつ学習 ===#
        for _, epoch in enumerate(prog_bar):

            #=== model を trainモード に変更
            self.train()

            #=== 記録用変数の初期化 ===#
            losses = 0.0
            correct = 0
            total = 0

            #=== 1エポックの学習 ===#
            for i, (_, inputs1, inputs2, targets) in enumerate(train_loader):
                
                # ----------------------------------------
                # ① 現在タスクのバッチを gpu に載せる
                # ----------------------------------------
                inputs1 = inputs1.to(self._device)
                inputs2 = inputs2.to(self._device)
                targets = targets.to(self._device)
                inputs = torch.cat([inputs1, inputs2], dim=0)
                targets = torch.cat([targets, targets], dim=0)

                # ----------------------------------------
                # ② Forward 処理
                # ----------------------------------------
                out = self._network(inputs)
                logits = out["logits"]
                embedding = out["embedding"]

                # ----------------------------------------
                # ③ 損失計算
                # ----------------------------------------
                loss_clf = F.cross_entropy(logits, targets)
                infonce_loss = infoNCE_loss(embedding, self.args['infonce_temp'])

                loss = loss_clf + infonce_loss * self.args['contrast_factor']

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
            self.train()

            #=== 記録用変数の初期化 ===#
            losses = 0.
            losses_clf = 0.
            losses_aux = 0.
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
                # ② 損失計算
                # ----------------------------------------
                logits, loss_clf, loss_aux, loss_forg = self.compute_loss(inputs, targets)
                loss = loss_clf * self.lambda_clf + loss_aux * self.lambda_aux +loss_forg * self.lambda_forg

                # ----------------------------------------
                # ③ 最適化処理
                # ----------------------------------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_aux += loss_aux.item()
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
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Loss_forg {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1, self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    losses_forg / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Loss_forg {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    losses_forg / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)
        logging.info(info)


    def compute_loss(self, inputs, targets):

        #=== model にサンプルを入力して outputs を取得 ===#
        outputs = self._network(inputs)

        logits = outputs["logits"]
        aux_logits = outputs["aux_logits"]

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

            loss_clf = F.cross_entropy(retain_logits, retain_targets)


        #=== aux 損失を計算 ===#
        loss_aux = torch.tensor(0., device=self._device)

        # 維持クラスがあれば損失計算
        if mask_retain.any():
            retain_aux_logits = aux_logits[mask_retain]

            aux_targets = retain_targets.clone()
            aux_targets = torch.where(
                aux_targets - self._known_classes + 1 > 0,
                aux_targets - self._known_classes + 1,
                0,
            )
            loss_aux = F.cross_entropy(retain_aux_logits, aux_targets)
        
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


        return logits ,loss_clf, loss_aux, loss_forg
    

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
        # NME 評価（DERでは無効化）
        # -------------------------
        nme_accy = None
        # logging.info(f"[Eval][Task {task_id}] NME disabled for DER (return None).")


        return cnn_accy, nme_accy





def infoNCE_loss(feats, t):
    cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll

def infoNCE_distill_loss(p_feats, z_feats, t):
    # print(p_feats.shape, z_feats.shape)
    cos_sim = F.cosine_similarity(p_feats[:,None,:], z_feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / t
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll








