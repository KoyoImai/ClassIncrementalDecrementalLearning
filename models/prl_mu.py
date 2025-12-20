
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
from utils.toolkit import count_parameters, target2onehot, tensor2numpy


import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module):

    def __init__(self):
        super(BaseAttention, self).__init__()

    def forward(self, x):
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x

class AutoencoderSigmoid(BaseAttention):
    def __init__(self, input_dims=512, code_dims=256):
        super(AutoencoderSigmoid, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dims),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(code_dims, input_dims),
            nn.Sigmoid())
    

class PES_Loss(nn.Module):
    def __init__(self, lamda = 0.5):
        super(PES_Loss, self).__init__()
        self.lamda = lamda

    def forward(self, features, labels=None):
        # device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        device = features.device

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  

        loss = (1.0 - pos_pairs_mean) + self.lamda * (1.0 + neg_pairs_mean)

        return loss



class PRLMU(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # モデルの作成
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

        # 学習ハイパラ（後続タスク）
        self.epochs = args["epochs"]
        self.lr = args["lr"]

        # 学習ハイパラ（共通）
        self.step_size = args["step_size"]
        self.gamma = args["gamma"]
        self.weight_decay = args["weight_decay"]
        self.lambda_clf = args["lambda_clf"]
        self.lambda_forg = args["lambda_forg"]

        # 忘却クラスのNME分類用
        self._forget_class_means = {}    # dict: class_id -> mean_vector (np.ndarray, shape [feature_dim])
        self._forget_class_targets = []  # list[int]

        # その他パラメータ
        self.num_workers = args["num_workers"]

        # プロトタイプの初期化
        self._protos = {}

        # PRL 特有の設定
        self.pes_loss_func = PES_Loss()
        self.old_ae = None

    def after_task(self, log_dir=None):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

        #=== 知識蒸留用の教師モデルを更新 ===#
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network

        #=== モデルの保存 ===#
        filename = f"{log_dir}phase"
        self.save_checkpoint(filename)

    #---------------------------------------------------------------------------------------
    # モデルの訓練
    #---------------------------------------------------------------------------------------
    def incremental_train(self, data_manager):
        
        #=== data manager の登録 ===#
        self.data_manager = data_manager

        #=== 現在タスクの更新 ===#
        self._cur_task += 1

        #=== 2タスク目は AutoEncoder を作成 ===#
        if self._cur_task == 1:
            self.old_ae = AutoencoderSigmoid(code_dims=512)
            self.old_ae.to(self._device)

        #=== 現在タスクのクラスまでを含めた合計のクラス数を更新 ===#
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        #=== 忘却クラスの更新 ===#
        self.forget_classes += [cls for cls in self.forget_list[self._cur_task]]
        logging.info("forget classes on task{}: {}".format(self._cur_task, self.forget_classes))

        #=== モデルの出力層を更新 ===#
        self._network.update_fc(self._total_classes*4)
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
            num_workers=self.num_workers,
            pin_memory=True
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
            num_workers=self.num_workers,
            pin_memory=True
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
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        
        #=== 1タスク目の学習 ===#
        if self._cur_task == 0:

            #=== Optimizer の設定 ===#
            optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )

            #=== Scheduler の設定 ===#
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=self.args["step_size"],
                gamma=self.args["gamma"]
            )
        #=== 2タスク目以降の学習 ===#
        else:

            #=== 最適化対象のパラメータ ===#
            trainable_list = nn.ModuleList([])
            trainable_list.append(self._network)
            trainable_list.append(self.old_ae)
            self._epoch_num = self.args["epochs"]
            logging.info('All params total: {}'.format(count_parameters(trainable_list)))

            #=== Optimizer の設定 ===#
            optimizer = torch.optim.Adam(
                trainable_list.parameters(),
                lr=self.args["lr"],
                weight_decay=self.args["weight_decay"]
            )

            #=== Scheduler の設定 ===#
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.args["step_size"], 
                amma=self.args["gamma"]
            )

        #=== 訓練を実行 ===#
        self._train_function(train_loader, test_loader, optimizer, scheduler)

        #=== プロトタイプを構築 ===#
        self._build_protos()


    def _train_function(self, train_loader, test_loader, optimizer, scheduler):

        #=== プログレスバーの設定 ===#
        prog_bar = tqdm(range(self.init_epoch))

        #=== 1エポックずつ学習 ===#
        for _, epoch in enumerate(prog_bar):

            #=== model を trainモード に変更
            self._network.train()

            #=== 記録用変数の初期化 ===#
            losses = 0.0
            losses_clf = 0.
            losses_fkd = 0.
            losses_proto = 0.
            losses_pes = 0.
            losses_pkd = 0.

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
                # ② class augmentaion の回転処理
                # ----------------------------------------
                inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
                inputs = inputs.view(-1, 3, self.size, self.size)

                aug_targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)

                # ----------------------------------------
                # ③ 損失計算
                # ----------------------------------------
                logits, loss_new, loss_fkd, loss_proto, loss_pes, loss_pkd, loss_unl = self._compute_loss(inputs, targets, aug_targets)
                loss = loss_new + loss_fkd + loss_proto + loss_pes + loss_pkd + loss_unl
    

    def _compute_loss(self, inputs, targets, aug_targets):

        pes_targets = torch.stack([targets for k in range(4)], 1).view(-1)

        # model に inputs を入力し特徴量を出力
        features = self._network_module_ptr.extract_vector(inputs)
        
        # 特徴量を fc層 に入力して logits を獲得
        logits = self._network_module_ptr.fc(features)["logits"]

        #=== 交差エントロピー損失を計算 ===#
        loss_clf = F.cross_entropy(logits/self.args["temp"], aug_targets)
        loss_new = loss_clf

        #=== PES 損失を計算 ===#
        if self._cur_task == 0:
            loss_iic = self.args["lambda_pes"] * self.pes_loss_func(features, pes_targets)

            loss_fkd = torch.tensor(0.)
            loss_proto = torch.tensor(0.)
            loss_pkd = torch.tensor(0.)
            loss_unl = torch.tensor(0.)

            return logits, loss_new, loss_fkd, loss_proto, loss_iic, loss_pkd, loss_unl

        #=== L2損失による蒸留損失 ===#
        loss_fkd = torch.tensor(0.)

        # 過去モデルの特徴量を取り出す
        features_old = self.old_network_module_ptr.extract_vector(inputs)

        loss_fkd = self.args["lambda_fkd"] * torch.dist(features, features_old, 2)

        #=== PGRU損失を計算 ===#
        loss_pkd = self.args["lambda_pgru"] * self._contras_loss(features, features_old)

        #=== プロトタイプ損失を計算 ===#
        loss_proto = torch.tensor(0.)

        # 擬似 feature ベクトルを追加するリスト
        proto_features = []
        
        # プロトタイプに対応したラベルを格納するリスト
        proto_targets = []

        # 旧タスクに存在するラベルのリスト
        old_class_list = list(self._protos.keys())

        # 旧タスクのラベルリストから忘却対象のクラスを除外する
        old_class_list = [c for c in old_class_list if c not in self.forget_classes]

        # バッチサイズ分だけサンプルを作成する
        for _ in range(features.shape[0]//4): # batch_size = feature.shape[0] // 4

            # ランダムでサンプルを1つ選択
            i = np.random.randint(0, features.shape[0])

            # 混ぜるプロトタイプをランダムに選択するためシャッフル
            np.random.shuffle(old_class_list)
            lam = np.random.beta(0.5, 0.5)
            if lam > 0.6:
                lam = lam * 0.6
            
            # サンプルの特徴とプロトタイプを mixup
            if np.random.random() >= 0.5:
                temp = (1 + lam) * self._protos[old_class_list[0]] - lam * features.detach().cpu().numpy()[i]
            else:
                temp = (1 - lam) * self._protos[old_class_list[0]] + lam * features.detach().cpu().numpy()[i]
            
            # 擬似サンプル（擬似特徴）をリストに格納
            proto_features.append(temp)

            # 擬似サンプルに対応するラベルを格納（ラベルはプロトタイプのラベル）
            proto_targets.append(old_class_list[0])
        
        proto_features = torch.from_numpy(np.asarray(proto_features)).float().to(self._device,non_blocking=True)
        proto_targets = torch.from_numpy(np.asarray(proto_targets)).to(self._device,non_blocking=True)
        
        proto_logits = self._network_module_ptr.fc(proto_features)["logits"]
        loss_proto = self.args["lambda_proto"] * F.cross_entropy(proto_logits/self.args["temp"], proto_targets*4)






    def _contras_loss(self, features, features_old):

        # 整合損失: AE（旧feature）と現在featureのMSE
        features_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        
        # 直交損失: AE(protos) と AE(旧feature) の cosine を下げる
        features_old_norm = F.normalize(features_old, p=2, dim=1)

        # 忘却クラスのプロトタイプの前準備
        valid_protos = [
            proto for cls_id, proto in self._protos.items()
            if cls_id not in self.forget_classes
        ]

        if len(valid_protos) == 0:
            # 直交させる相手がいないので align 項だけにする
            return loss_align

        protos = np.asarray(valid_protos)  # shape: [num_valid_classes, D]
        protos = torch.from_numpy(protos).float().to(self._device, non_blocking=True)
        protos = self.old_ae(protos)       # AutoEncoderで射影
        protos = F.normalize(protos, p=2, dim=1)

        # # プロトタイプの準備（not Machine Unlearning用）
        # protos = self._protos.values()             # 各クラスのプロトタイプ
        # protos = torch.from_numpy(np.asarray(list(protos))).float().to(self._device,non_blocking=True)
        # protos = self.old_ae(protos)               # AutoEncoderで射影
        # protos = F.normalize(protos, p=2, dim=1)

        similarity = torch.matmul(protos, features_old_norm.t())
        similarity = similarity.sum() / (similarity.shape[0]*similarity.shape[1])
        
        return loss_align + similarity


