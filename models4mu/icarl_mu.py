import os
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
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

from utils.grad_conflict import GradConflictLogger, select_named_params

EPSILON = 1e-8



class iCaRLMU(BaseLearner):
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

        # 保存するパラメータ
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
            "forget_classes": copy.deepcopy(getattr(self, "forget_classes", [])),
        }

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







