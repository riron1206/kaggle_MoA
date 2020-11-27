import sys

# sys.path.append("../input/iterative-stratification/iterative-stratification-master")
sys.path.append(r"C:\Users\81908\Git\iterative-stratification")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

sys.path.append(
    r"C:\Users\81908\Git\Adabelief-Optimizer\pypi_packages\adabelief_pytorch0.1.0"
)
from adabelief_pytorch import AdaBelief

# + _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0"
import os
import copy
import random
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
    QuantileTransformer,
    OneHotEncoder,
)
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import time
import warnings

warnings.filterwarnings("ignore")
# -

# # Parameters

# +
# train_features = pd.read_csv("../input/lish-moa/train_features.csv")
# train_targets_scored = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
# train_targets_nonscored = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")
#
# test_features = pd.read_csv("../input/lish-moa/test_features.csv")
# sample_submission = pd.read_csv("../input/lish-moa/sample_submission.csv")
#
# train_drug = pd.read_csv("../input/lish-moa/train_drug.csv")

# +
# dtype = {"cp_type": "category", "cp_dose": "category"}
# index_col = "sig_id"

sys.path.append(r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\code")
import datasets

DATADIR = datasets.DATADIR

train_features = pd.read_csv(
    f"{DATADIR}/train_features.csv"
    # , dtype=dtype, index_col=index_col
)
X = train_features.select_dtypes("number")
train_targets_nonscored = pd.read_csv(
    f"{DATADIR}/train_targets_nonscored.csv"
    # , index_col=index_col
)
train_targets_scored = pd.read_csv(
    f"{DATADIR}/train_targets_scored.csv"
    # , index_col=index_col
)
train_drug = pd.read_csv(
    f"{DATADIR}/train_drug.csv"
    # , dtype=dtype, index_col=index_col, squeeze=True
)

test_features = pd.read_csv(f"{DATADIR}/test_features.csv")
sample_submission = pd.read_csv(f"{DATADIR}/sample_submission.csv")

columns = train_targets_scored.iloc[:, 1:].columns

# +
params = {
    "n_genes_pca": 50,
    "n_cells_pca": 20,
    "batch_size": 256,
    # "model": "MLP_2HL_weight_norm",
    "model": "MLP_2HL",
    # "optimizer": "adabelief",
    "optimizer": "adam",
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "n_folds": 5,
    "early_stopping_steps": 5,
    "hidden_size": 512,
    "boost_rate": 1.0,  # original: 1.0
    "num_nets": 20,  # Number of weak NNs. original: 40 n_estimators?
    "epochs_per_stage": 1,  # Number of epochs to learn the Kth model. original: 1
    "correct_epoch": 1,  # Number of epochs to correct the whole week models original: 1
    "model_order": "second",  # You could put "first" according to the original implemention, but error occurs. original: "second"
}
n_seeds = 5
# -

GENES = [col for col in train_features.columns if col.startswith("g-")]
CELLS = [col for col in train_features.columns if col.startswith("c-")]


# + code_folding=[0]
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)
# -

# # Preprocessing

# + code_folding=[4]
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class ClippedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True, high=0.99, low=0.01):
        self.copy = copy
        self.high = high
        self.low = low

    def fit(self, X, y=None):
        self.data_max_ = X.quantile(q=self.high)
        self.data_min_ = X.quantile(q=self.low)

        return self

    def transform(self, X):
        if self.copy:
            X = X.copy()

        X.clip(self.data_min_, self.data_max_, axis=1, inplace=True)

        return X


# +
clipped_features = ClippedFeatures()
X = clipped_features.fit_transform(X)

with open("clipped_features.pkl", "wb") as f:
    pickle.dump(clipped_features, f)

train_features[X.columns] = X

# +
feature_cols = train_features.columns[4:].tolist()

feature_cols = ["cp_time"] + train_features.columns[4:].tolist()

params["feat_d"] = len(feature_cols)

# +
train = train_features.merge(train_targets_scored, on="sig_id")
# train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)

test = test_features.copy()
# test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]
# -

train = train.drop("cp_type", axis=1)
test = test.drop("cp_type", axis=1)


if "sig_id" in target.columns:
    target_cols = target.drop("sig_id", axis=1).columns.values.tolist()
else:
    target_cols = target.columns.values.tolist()

# # CV strategy

# +
folds_ = train.copy()

folds = []

# LOAD FILES
train_feats = train_features
scored = target
drug = train_drug
targets = target_cols
scored = scored.merge(drug, on="sig_id", how="left")

# LOCATE DRUGS
vc = scored.drug_id.value_counts()
vc1 = vc.loc[(vc == 6) | (vc == 12) | (vc == 18)].index.sort_values()
vc2 = vc.loc[(vc != 6) & (vc != 12) & (vc != 18)].index.sort_values()

# STRATIFY DRUGS 18X OR LESS
dct1 = {}
dct2 = {}
skf = MultilabelStratifiedKFold(
    n_splits=params["n_folds"], shuffle=True, random_state=0
)
tmp = scored.groupby("drug_id")[targets].mean().loc[vc1]
for fold, (idxT, idxV) in enumerate(skf.split(tmp, tmp[targets])):
    dd = {k: fold for k in tmp.index[idxV].values}
    dct1.update(dd)

# STRATIFY DRUGS MORE THAN 18X
skf = MultilabelStratifiedKFold(
    n_splits=params["n_folds"], shuffle=True, random_state=0
)
tmp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop=True)
for fold, (idxT, idxV) in enumerate(skf.split(tmp, tmp[targets])):
    dd = {k: fold for k in tmp.sig_id[idxV].values}
    dct2.update(dd)

# ASSIGN FOLDS
scored["fold"] = scored.drug_id.map(dct1)
scored.loc[scored.fold.isna(), "fold"] = scored.loc[scored.fold.isna(), "sig_id"].map(
    dct2
)
scored.fold = scored.fold.astype("int8")
folds.append(scored.fold.values)

del scored["fold"]

s = np.stack(folds)
train["kfold"] = s.reshape(-1,)

# mskf = MultilabelStratifiedKFold(n_splits=5)


# -

# # Dataset Classes

# + code_folding=[0, 16]
class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        dct = {
            "x": torch.tensor(self.features[idx, :], dtype=torch.float),
            "y": torch.tensor(self.targets[idx, :], dtype=torch.float),
        }
        return dct


class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        dct = {"x": torch.tensor(self.features[idx, :], dtype=torch.float)}
        return dct


# -

# # Dynamic Model

# + code_folding=[3, 10]
from enum import Enum


class ForwardType(Enum):
    SIMPLE = 0
    STACKED = 1
    CASCADE = 2
    GRADIENT = 3


class DynamicNet(object):
    def __init__(self, c0, lr):
        self.models = []
        self.c0 = c0
        self.lr = lr
        self.boost_rate = nn.Parameter(
            torch.tensor(lr, requires_grad=True, device=device)
        )

    def add(self, model):
        self.models.append(model)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        params.append(self.boost_rate)
        return params

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_cuda(self):
        for m in self.models:
            m.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def forward(self, x):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0 = np.repeat(self.c0.detach().cpu().numpy().reshape(1, -1), batch, axis=0)
            c0 = torch.Tensor(c0).cuda() if device == "cuda" else torch.Tensor(c0)
            return None, c0
        middle_feat_cum = None
        prediction = None
        with torch.no_grad():
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum, prediction = m(x, middle_feat_cum)
                else:
                    middle_feat_cum, pred = m(x, middle_feat_cum)
                    prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    def forward_grad(self, x):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0 = np.repeat(self.c0.detach().cpu().numpy().reshape(1, -1), batch, axis=0)
            return None, torch.Tensor(c0).cuda()
        # at least one model
        middle_feat_cum = None
        prediction = None
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, prediction = m(x, middle_feat_cum)
            else:
                middle_feat_cum, pred = m(x, middle_feat_cum)
                prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    @classmethod
    def from_file(cls, path, builder):
        d = torch.load(path)
        net = DynamicNet(d["c0"], d["lr"])
        net.boost_rate = d["boost_rate"]
        for stage, m in enumerate(d["models"]):
            submod = builder(stage)
            submod.load_state_dict(m)
            net.add(submod)
        return net

    def to_file(self, path):
        models = [m.state_dict() for m in self.models]
        d = {
            "models": models,
            "c0": self.c0,
            "lr": self.lr,
            "boost_rate": self.boost_rate,
        }
        torch.save(d, path)


# -

# # Weak Models

# + code_folding=[0, 29]
# class MLP_1HL(nn.Module):
#    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
#        super(MLP_1HL, self).__init__()
#        self.layer1 = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim_in, dim_hidden1),)
#        self.layer2 = nn.Sequential(nn.ReLU(), nn.Linear(dim_hidden1, len(targets)),)
#        if bn:
#            self.bn = nn.BatchNorm1d(dim_hidden1)
#            self.bn2 = nn.BatchNorm1d(dim_in)
#
#    def forward(self, x, lower_f):
#        if lower_f is not None:
#            x = torch.cat([x, lower_f], dim=1)
#            x = self.bn2(x)
#        out = self.layer1(x)
#        return out, self.layer2(out)
#
#    @classmethod
#    def get_model(cls, stage, params):
#        if stage == 0:
#            dim_in = params["feat_d"]
#        else:
#            dim_in = params["feat_d"] + params["hidden_size"]
#        model = MLP_1HL(dim_in, params["hidden_size"], params["hidden_size"])
#        return model


class MLP_2HL(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP_2HL, self).__init__()
        self.bn2 = nn.BatchNorm1d(dim_in)

        self.layer1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim_in, dim_hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(dim_hidden1),
            nn.Dropout(0.4),
            nn.Linear(dim_hidden1, dim_hidden2),
        )
        self.layer2 = nn.Sequential(nn.ReLU(), nn.Linear(dim_hidden2, len(targets)),)

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        middle_feat = self.layer1(x)
        out = self.layer2(middle_feat)
        return middle_feat, out

    @classmethod
    def get_model(cls, stage, params):
        if stage == 0:
            dim_in = params["feat_d"]
        else:
            dim_in = params["feat_d"] + params["hidden_size"]
        model = MLP_2HL(dim_in, params["hidden_size"], params["hidden_size"])
        return model


class MLP_2HL_weight_norm(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP_2HL_weight_norm, self).__init__()
        self.bn2 = nn.BatchNorm1d(dim_in)

        self.layer1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.utils.weight_norm(nn.Linear(dim_in, dim_hidden1)),
            nn.ReLU(),
            nn.BatchNorm1d(dim_hidden1),
            nn.Dropout(0.4),
            nn.utils.weight_norm(nn.Linear(dim_hidden1, dim_hidden2)),
        )
        self.layer2 = nn.Sequential(
            nn.ReLU(), nn.utils.weight_norm(nn.Linear(dim_hidden2, len(targets))),
        )

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        middle_feat = self.layer1(x)
        out = self.layer2(middle_feat)
        return middle_feat, out

    @classmethod
    def get_model(cls, stage, params):
        if stage == 0:
            dim_in = params["feat_d"]
        else:
            dim_in = params["feat_d"] + params["hidden_size"]
        model = MLP_2HL_weight_norm(
            dim_in, params["hidden_size"], params["hidden_size"]
        )
        return model


class MLP_2HL_leaky_relu(nn.Module):
    # https://www.kaggle.com/vbmokin/moa-pytorch-rankgauss-pca-nn-upgrade-3d-visual
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP_2HL_leaky_relu, self).__init__()
        self.bn2 = nn.BatchNorm1d(dim_in)

        self.dense1 = nn.utils.weight_norm(nn.Linear(dim_in, dim_hidden1))

        self.batch_norm2 = nn.BatchNorm1d(dim_hidden1)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.utils.weight_norm(nn.Linear(dim_hidden1, dim_hidden2))

        self.batch_norm3 = nn.BatchNorm1d(dim_hidden2)
        self.dropout3 = nn.Dropout(0.4)
        self.dense3 = nn.utils.weight_norm(nn.Linear(dim_hidden2, len(targets)))

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)

        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        middle_feat = self.dropout3(x)
        out = self.dense3(middle_feat)

        return middle_feat, out

    @classmethod
    def get_model(cls, stage, params):
        if stage == 0:
            dim_in = params["feat_d"]
        else:
            dim_in = params["feat_d"] + params["hidden_size"]
        model = MLP_2HL_leaky_relu(dim_in, params["hidden_size"], params["hidden_size"])
        return model


# + code_folding=[3]
from torch.nn.modules.loss import _WeightedLoss


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


# + code_folding=[0, 6]
def get_optim(params, lr, weight_decay):
    optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
    # optimizer = SGD(params, lr, weight_decay=weight_decay)
    return optimizer


def get_optim_adabelief(params, lr, weight_decay):
    optimizer = AdaBelief(
        params,
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-16,
        betas=(0.9, 0.999),
        weight_decouple=True,
        rectify=False,
    )
    return optimizer


def logloss(net_ensemble, test_loader):
    loss = 0
    total = 0
    loss_f = (
        nn.BCEWithLogitsLoss()
    )  # Binary cross entopy loss with logits, reduction=mean by default
    for data in test_loader:
        x = data["x"].cuda() if device == "cuda" else data["x"]
        y = data["y"].cuda() if device == "cuda" else data["y"]
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        loss += loss_f(out, y)
        total += 1

    return loss / total


# -

device = "cuda" if torch.cuda.is_available() else "cpu"

# # Training

# + code_folding=[]
c0_ = np.log(np.mean(train_targets_scored.iloc[:, 1:].values, axis=0))


def train_fn(n_seeds):
    print(f"params: {params}")

    oof = np.zeros((len(train), len(target_cols)))
    # predictions = np.zeros((len(test), len(target_cols)))

    for seed in tqdm(range(n_seeds)):
        seed_everything(seed)
        print("-" * 100)

        for fold in range(params["n_folds"]):
            print("=" * 25, f"fold: {fold}", "=" * 25)

            train_idx = train[train["kfold"] != fold].index
            val_idx = train[train["kfold"] == fold].index

            train_df = train[train["kfold"] != fold].reset_index(drop=True)
            val_df = train[train["kfold"] == fold].reset_index(drop=True)

            x_train = train_df[feature_cols].values
            y_train = train_df[target_cols].values

            x_val = val_df[feature_cols].values
            y_val = val_df[target_cols].values

            train_ds = MoADataset(x_train, y_train)
            val_ds = MoADataset(x_val, y_val)
            train_loader = DataLoader(
                train_ds, batch_size=params["batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=params["batch_size"], shuffle=False
            )

            best_score = np.inf
            val_score = best_score
            best_stage = params["num_nets"] - 1

            c0 = torch.tensor(c0_, dtype=torch.float).to(device)
            net_ensemble = DynamicNet(c0, params["boost_rate"])
            loss_f1 = nn.MSELoss(reduction="none")
            loss_f2 = SmoothBCEwLogits(smoothing=0.001, reduction="none")
            # loss_f2 = nn.BCEWithLogitsLoss(reduction='none')
            loss_models = torch.zeros((params["num_nets"], 3))

            all_ensm_losses = []
            all_ensm_losses_te = []
            all_mdl_losses = []
            dynamic_br = []

            lr = params["lr"]
            L2 = params["weight_decay"]

            early_stop = 0
            for stage in range(params["num_nets"]):
                t0 = time.time()

                if params["model"] == "MLP_2HL_weight_norm":
                    model = MLP_2HL_weight_norm.get_model(stage, params)
                elif params["model"] == "MLP_2HL_leaky_relu":
                    model = MLP_2HL_leaky_relu.get_model(stage, params)
                else:
                    model = MLP_2HL.get_model(stage, params)
                model.to(device)

                if params["optimizer"] == "adam":
                    optimizer = get_optim(model.parameters(), lr, L2)
                elif params["optimizer"] == "adabelief":
                    optimizer = get_optim_adabelief(model.parameters(), lr, L2)

                net_ensemble.to_train()  # Set the models in ensemble net to train mode
                stage_mdlloss = []
                for epoch in range(params["epochs_per_stage"]):
                    for i, data in enumerate(train_loader):
                        x = data["x"].to(device)
                        y = data["y"].to(device)
                        middle_feat, out = net_ensemble.forward(x)
                        if params["model_order"] == "first":
                            grad_direction = y / (1.0 + torch.exp(y * out))
                        else:
                            h = 1 / (
                                (1 + torch.exp(y * out)) * (1 + torch.exp(-y * out))
                            )
                            grad_direction = y * (1.0 + torch.exp(-y * out))
                            nwtn_weights = (torch.exp(out) + torch.exp(-out)).abs()
                        _, out = model(x, middle_feat)
                        loss = loss_f1(net_ensemble.boost_rate * out, grad_direction)
                        loss = loss * h
                        loss = loss.mean()
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()
                        stage_mdlloss.append(loss.item())

                net_ensemble.add(model)
                sml = np.mean(stage_mdlloss)

                stage_loss = []
                lr_scaler = 2
                # fully-corrective step
                if stage != 0:
                    # Adjusting corrective step learning rate
                    if stage % 3 == 0:
                        lr /= 2
                    optimizer = get_optim(net_ensemble.parameters(), lr / lr_scaler, L2)
                    for _ in range(params["correct_epoch"]):
                        for i, data in enumerate(train_loader):
                            x = data["x"].to(device)
                            y = data["y"].to(device)
                            _, out = net_ensemble.forward_grad(x)
                            loss = loss_f2(out, y).mean()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            stage_loss.append(loss.item())

                sl_te = logloss(net_ensemble, val_loader)

                # Store dynamic boost rate
                dynamic_br.append(net_ensemble.boost_rate.item())

                elapsed_tr = time.time() - t0
                sl = 0
                if stage_loss != []:
                    sl = np.mean(stage_loss)

                all_ensm_losses.append(sl)
                all_ensm_losses_te.append(sl_te)
                all_mdl_losses.append(sml)
                print(
                    f"Stage - {stage}, training time: {elapsed_tr: .1f} sec, boost rate: {net_ensemble.boost_rate: .4f}, Training Loss: {sl: .5f}, Val Loss: {sl_te: .5f}"
                )

                if device == "cuda":
                    net_ensemble.to_cuda()
                net_ensemble.to_eval()  # Set the models in ensemble net to eval mode

                # Train
                # print('Acc results from stage := ' + str(stage) + '\n')
                # AUC
                # val_score = auc_score(net_ensemble, val_loader)
                if sl_te < best_score:
                    best_score = sl_te
                    best_stage = stage
                    net_ensemble.to_file(f"./{fold}FOLD_{seed}_.pth")
                    early_stop = 0
                else:
                    early_stop += 1

                # test_score = auc_score(net_ensemble, val_loader)
                # print(f'Stage: {stage}, AUC@Val: {val_score:.4f}, AUC@Test: {test_score:.4f}')
                # loss_models[stage, 1], loss_models[stage, 2] = val_score, test_score

                if early_stop > params["early_stopping_steps"]:
                    print("early stopped!")
                    break

            # val_auc, te_auc = loss_models[best_stage, 1], loss_models[best_stage, 2]
            print(f"Best validation stage: {best_stage}")

            if params["model"] == "MLP_2HL_weight_norm":
                net_ensemble = DynamicNet.from_file(
                    f"./{fold}FOLD_{seed}_.pth",
                    lambda stage: MLP_2HL_weight_norm.get_model(stage, params),
                )
            elif params["model"] == "MLP_2HL_leaky_relu":
                net_ensemble = DynamicNet.from_file(
                    f"./{fold}FOLD_{seed}_.pth",
                    lambda stage: MLP_2HL_leaky_relu.get_model(stage, params),
                )
            else:
                net_ensemble = DynamicNet.from_file(
                    f"./{fold}FOLD_{seed}_.pth",
                    lambda stage: MLP_2HL.get_model(stage, params),
                )
            if device == "cuda":
                net_ensemble.to_cuda()
            net_ensemble.to_eval()

            # --------------------- PREDICTION---------------------

            preds = []
            with torch.no_grad():
                for data in val_loader:
                    x = data["x"].to(device)
                    _, pred = net_ensemble.forward(x)
                    preds.append(pred.sigmoid().detach().cpu().numpy())
            oof[val_idx, :] += np.concatenate(preds) / n_seeds

            # x_test = test[feature_cols].values
            # test_ds = TestDataset(x_test)
            # test_loader = DataLoader(
            #    test_ds, batch_size=params["batch_size"], shuffle=False
            # )
            # preds = []
            # with torch.no_grad():
            #    for data in test_loader:
            #        x = data["x"].to(device)
            #        _, pred = net_ensemble.forward(x)
            #        preds.append(pred.sigmoid().detach().cpu().numpy())
            # predictions += np.concatenate(preds) / (params["n_folds"] * n_seeds)

    # 予測値クリップ
    # oof = np.clip(oof, 1e-3, 1 - 1e-3)
    # predictions = np.clip(predictions, 1e-3, 1 - 1e-3)

    train[target_cols] = oof
    # test[target_cols] = predictions

    val_results = (
        train_targets_scored.drop(columns=target_cols)
        .merge(train[["sig_id"] + target_cols], on="sig_id", how="left")
        .fillna(0)
    )

    y_true = train_targets_scored[target_cols].values
    y_pred = val_results[target_cols].values

    score = 0
    for i in range(len(target_cols)):
        score_ = log_loss(y_true[:, i], y_pred[:, i])
        score += score_ / len(target_cols)
    print("CV log_loss ", score)

    # sub = sample_submission
    # sub = (
    #    sub.drop(columns=target_cols)
    #    .merge(test[["sig_id"] + target_cols], on="sig_id", how="left")
    #    .fillna(0)
    # )

    # sub = sub.drop(columns=self.target_cols).merge(test_[["sig_id"]+self.target_cols+["cp_time_24", "cp_dose_D2"]], on="sig_id", how="left").fillna(0)
    # sub.loc[:, ["atp-sensitive_potassium_channel_antagonist", "erbb2_inhibitor"]] = 0.000012
    # sub = sub.drop(["cp_time_24", "cp_dose_D2"], axis=1)

    y_pred = pd.DataFrame(y_pred, index=train["sig_id"], columns=target_cols)
    with open("Y_pred.pkl", "wb") as f:
        pickle.dump(y_pred, f)

    return score, y_pred
