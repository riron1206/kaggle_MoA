# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import gc
import re
import math
import pickle
import joblib
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

warnings.simplefilter("ignore")

# + _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a"
import os
import random as rn
import numpy as np


def set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)

    rn.seed(seed)
    np.random.seed(seed)


# +
from sklearn.metrics import log_loss


def score(Y, Y_pred):
    _, n_classes = Y.shape

    losses = []

    for j in range(n_classes):
        loss = log_loss(Y.iloc[:, j], Y_pred.iloc[:, j], labels=[0, 1])

        losses.append(loss)

    return np.mean(losses)


# +
import sys

# sys.path.append('../input/iterativestratification')

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection._split import _BaseKFold


class MultilabelGroupStratifiedKFold(_BaseKFold):
    def __init__(self, n_splits=5, random_state=None, shuffle=False):
        super().__init__(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        cv = MultilabelStratifiedKFold(
            n_splits=self.n_splits,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        value_counts = groups.value_counts()
        regular_index = value_counts.loc[
            (value_counts == 6) | (value_counts == 12) | (value_counts == 18)
        ].index.sort_values()
        irregular_index = value_counts.loc[
            (value_counts != 6) & (value_counts != 12) & (value_counts != 18)
        ].index.sort_values()

        group_to_fold = {}
        tmp = Y.groupby(groups).mean().loc[regular_index]

        for fold, (_, test) in enumerate(cv.split(tmp, tmp)):
            group_to_fold.update({group: fold for group in tmp.index[test]})

        sample_to_fold = {}
        tmp = Y.loc[groups.isin(irregular_index)]

        for fold, (_, test) in enumerate(cv.split(tmp, tmp)):
            sample_to_fold.update({sample: fold for sample in tmp.index[test]})

        folds = groups.map(group_to_fold)
        is_na = folds.isna()
        folds[is_na] = folds[is_na].index.map(sample_to_fold).values

        for i in range(self.n_splits):
            yield np.where(folds == i)[0]


# +
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
import pandas as pd


def compute_row_statistics(X, prefix=""):
    Xt = pd.DataFrame()

    for agg_func in [
        # "min",
        # "max",
        "mean",
        "std",
        "kurtosis",
        "skew",
    ]:
        Xt[f"{prefix}{agg_func}"] = X.agg(agg_func, axis=1)

    return Xt


# +
import seaborn as sns
import matplotlib.pyplot as plt


def display_importances(
    importance_df, png_path=f"feature_importance.png",
):
    """feature_importance plot"""
    importance_df.sort_values(by="importance", ascending=False).to_csv(
        f"feature_importance.csv"
    )
    cols = (
        importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:100]
        .index
    )
    best_features = importance_df.loc[importance_df.feature.isin(cols)]
    plt.figure(figsize=(8, 15))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False),
    )
    plt.title("LightGBM (avg over folds)")
    plt.tight_layout()
    plt.savefig(png_path)


# +
# dtype = {"cp_type": "category", "cp_dose": "category"}
# index_col = "sig_id"
#
# train_features = pd.read_csv(
#    "../input/lish-moa/train_features.csv", dtype=dtype, index_col=index_col
# )
# X = train_features.select_dtypes("number")
# Y_nonscored = pd.read_csv(
#    "../input/lish-moa/train_targets_nonscored.csv", index_col=index_col
# )
# Y = pd.read_csv("../input/lish-moa/train_targets_scored.csv", index_col=index_col)
# groups = pd.read_csv(
#    "../input/lish-moa/train_drug.csv", index_col=index_col, squeeze=True
# )
#
# columns = Y.columns

# +
DATADIR = (
    r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\input\lish-moa"
)

dtype = {"cp_type": "category", "cp_dose": "category"}
index_col = "sig_id"

train_features = pd.read_csv(
    f"{DATADIR}/train_features.csv", dtype=dtype, index_col=index_col
)
X = train_features.select_dtypes("number")
Y_nonscored = pd.read_csv(f"{DATADIR}/train_targets_nonscored.csv", index_col=index_col)
Y = pd.read_csv(f"{DATADIR}/train_targets_scored.csv", index_col=index_col)
groups = pd.read_csv(f"{DATADIR}/train_drug.csv", index_col=index_col, squeeze=True)

columns = Y.columns

# +
clipped_features = ClippedFeatures()
X = clipped_features.fit_transform(X)

with open("clipped_features.pkl", "wb") as f:
    pickle.dump(clipped_features, f)
# アンサンブルのために統計値, nonscoredは入れない
# c_prefix = "c-"
# g_prefix = "g-"
# c_columns = X.columns.str.startswith(c_prefix)
# g_columns = X.columns.str.startswith(g_prefix)
# X_c = compute_row_statistics(X.loc[:, c_columns], prefix=c_prefix)
# X_g = compute_row_statistics(X.loc[:, g_columns], prefix=g_prefix)
# X = pd.concat([X, X_c, X_g], axis=1)
# -

# # objective


def train_and_evaluate(params):

    _count = 0
    counts = np.empty((n_seeds * len(columns) * n_splits))

    f_importance = np.zeros((n_features,))
    Y_pred = np.zeros((train_size, n_classes))
    Y_pred = pd.DataFrame(Y_pred, columns=Y.columns, index=Y.index)

    for i in range(n_seeds):
        set_seed(seed=i)

        for tar_i, tar_col in tqdm(enumerate(Y.columns)):
            Y_target = Y[[tar_col]]

            if is_drug_cv:
                cv = MultilabelGroupStratifiedKFold(
                    n_splits=n_splits, random_state=i, shuffle=True
                )
                cv_split = cv.split(X, Y_target, groups)
            else:
                StratifiedKFold(n_splits=n_splits, random_state=i, shuffle=True)
                cv_split = cv.split(X, Y_target)

            for j, (trn_idx, val_idx) in enumerate(cv_split):
                print(f"\n------------ fold:{j} ------------")
                counts[_count] = Y_target.iloc[trn_idx].sum()

                X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
                Y_train, Y_val = Y_target.iloc[trn_idx], Y_target.iloc[val_idx]

                # Label Smoothing. https://www.kaggle.com/gogo827jz/self-stacking-groupcv-xgboost
                Y_train = Y_train * (1 - LBS) + 0.5 * LBS

                lgb_train = lgb.Dataset(X_train, Y_train)
                lgb_eval = lgb.Dataset(X_val, Y_val, reference=lgb_train)

                model = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_eval],
                    verbose_eval=verbose_eval,
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                )
                Y_pred[tar_col][val_idx] += (
                    model.predict(X_val, num_iteration=model.best_iteration) / n_seeds
                )

                # f_importance += np.array(model.feature_importance(importance_type="gain")) / (n_seeds * n_splits)

                joblib.dump(
                    model,
                    f"model_seed_{i}_fold_{j}_{Y.columns[tar_i]}.jlb",
                    compress=True,
                )
                _count += 1

    # importance_df = pd.DataFrame({"feature": model.feature_name(), "importance": f_importance})

    Y_pred[train_features["cp_type"] == "ctl_vehicle"] = 0.0

    with open("counts.pkl", "wb") as f:
        pickle.dump(counts, f)

    with open("Y_pred.pkl", "wb") as f:
        pickle.dump(Y_pred[columns], f)

    oof = score(Y[columns], Y_pred[columns])

    return oof, Y_pred


# +
import optuna


def objective(trial):
    params = {
        "objective": "binary",
        "learning_rate": 0.1,
    }
    params["max_depth"] = trial.suggest_int("max_depth", 1, 3)
    params["num_leaves"] = trial.suggest_int("num_leaves", 2, 4)
    params["min_data_in_leaf"] = trial.suggest_int(
        "min_data_in_leaf",
        1,
        max(1, int(X.shape[0] * ((n_splits - 1) / n_splits) / params["num_leaves"])),
    )
    params["feature_fraction"] = trial.suggest_discrete_uniform(
        "feature_fraction", 0.1, 1.0, 0.05
    )
    params["lambda_l1"] = trial.suggest_loguniform("lambda_l1", 1e-09, 10.0)
    params["lambda_l2"] = trial.suggest_loguniform("lambda_l2", 1e-09, 10.0)

    if DEBUG:
        params["n_estimators"] = 2
    # else:
    #    params["n_estimators"] = 1000

    oof, _ = train_and_evaluate(params)

    return oof


# +
is_drug_cv = True
n_splits = 5
n_seeds = 1
# LBS = 0.0008  # ラベルスムージングは全然効かないからやめる
LBS = 0.0

n_trials = 50
# params = {
#    "num_leaves": 2,
#    "max_depth": 1,
#    "min_data_in_leaf": 969,
#    "objective": "binary",
#    "learning_rate": 0.01,
# }
num_boost_round = 2000
verbose_eval = 100
# verbose_eval = 0  # 0なら学習履歴出さないが、warningは出るので意味なし
# num_boost_round = 50000
# verbose_eval = 1000
early_stopping_rounds = 100

# DEBUG = True
DEBUG = False
if DEBUG:
    columns = [
        "atp-sensitive_potassium_channel_antagonist",  # 陽性ラベル1個だけ
        "erbb2_inhibitor",  # 陽性ラベル1個だけ
        "antiarrhythmic",  # 陽性ラベル6個だけ
        #        "aldehyde_dehydrogenase_inhibitor",  # 陽性ラベル7個だけ
        #        "lipase_inhibitor",  # 陽性ラベル12個だけ
        #        "sphingosine_receptor_agonist",  # 陽性ラベル25個だけ
        #        "igf-1_inhibitor",  # 陽性ラベル37個だけ
        #        "potassium_channel_activator",  # 陽性ラベル55個だけ
        #        "potassium_channel_antagonist",  # 陽性ラベル98個だけ
        #        "dopamine_receptor_agonist",  # 陽性ラベル121個だけ
        #        "nfkb_inhibitor",  # 陽性ラベル832個
        #        "cyclooxygenase_inhibitor",  # 陽性ラベル435個
        #        "dna_inhibitor",  # 陽性ラベル402個
        #        "glutamate_receptor_antagonist",  # 陽性ラベル367個
        #        "tubulin_inhibitor",  # 陽性ラベル316個
        #        "pdgfr_inhibitor",  # 陽性ラベル297個
        #        "calcium_channel_blocker",  # 陽性ラベル281個
        #        "flt3_inhibitor",  # 陽性ラベル279個
        #        "progesterone_receptor_agonist",  # 陽性ラベル119個
        #        "hdac_inhibitor",  # 陽性ラベル106個
    ]
    Y = Y[columns]

    n_trials = 3
    # params["n_estimators"] = 2
    n_splits = 2
    num_boost_round = 100
    verbose_eval = num_boost_round
    early_stopping_rounds = verbose_eval
    print(f"DEBUG: {DEBUG}")
# -

train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

# +
# %%time

study = optuna.create_study(
    study_name="study",
    storage=f"sqlite:///study.db",
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=1),
)
study.optimize(objective, n_trials=n_trials)
study.trials_dataframe().to_csv(f"objective_history.csv", index=False)
with open(f"objective_best_params.txt", mode="w") as f:
    f.write(str(study.best_params))
print(f"\nstudy.best_params:\n{study.best_params}")

# +
params = study.best_params
params["objective"] = "binary"
params["learning_rate"] = 0.01
# params["n_estimators"] = 100  # default param

oof, Y_pred = train_and_evaluate(params)
print(oof)
# -

# # Platt Scaling
# Train a Logistic Regression model to calibrate the results
# - https://www.kaggle.com/gogo827jz/kernel-logistic-regression-one-for-206-targets

# +
# predict_probaでだしたY_predをロジスティク回帰で確率に補正する
# （Sigmoid関数にフィットさせ、そのSigmoid関数に通した値をCalibrationした値とする）

counts = np.empty((n_classes))

X_new = Y_pred.values
Y_cali = Y_pred.copy()

for tar in tqdm(range(Y.shape[1])):

    targets = Y.values[:, tar]
    X_targets = X_new[:, tar]
    counts[tar] = targets.sum()

    if targets.sum() >= n_splits:

        Y_cali[Y.columns[tar]] = np.zeros((Y_cali.shape[0],))

        skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)

        for n, (tr, te) in enumerate(skf.split(targets, targets)):
            x_tr, x_val = X_targets[tr].reshape(-1, 1), X_targets[te].reshape(-1, 1)
            y_tr, y_val = targets[tr], targets[te]

            model = LogisticRegression(penalty="none", max_iter=1000)
            model.fit(x_tr, y_tr)
            Y_cali[Y.columns[tar]].iloc[te] += model.predict_proba(x_val)[:, 1]

            joblib.dump(
                model, f"calibrate_model_target_{Y.columns[tar]}.jlb", compress=True
            )

with open("counts_calibrate.pkl", "wb") as f:
    pickle.dump(counts, f)

with open("Y_pred_calibrate.pkl", "wb") as f:
    pickle.dump(Y_cali[columns], f)

# -

score(Y[columns], Y_cali[columns])

# # pkl check

path = r"counts.pkl"
with open(path, "rb") as f:
    counts = pickle.load(f)
counts

path = r"counts_calibrate.pkl"
with open(path, "rb") as f:
    counts = pickle.load(f)
counts

path = r"Y_pred.pkl"
with open(path, "rb") as f:
    Y_pred = pickle.load(f)
Y_pred

path = r"Y_pred_calibrate.pkl"
with open(path, "rb") as f:
    Y_pred = pickle.load(f)
Y_pred

# # predict test

# +
import glob
import pathlib

test_features = pd.read_csv(
    # "../input/lish-moa/test_features.csv", dtype=dtype, index_col=index_col
    f"{DATADIR}/test_features.csv",
    dtype=dtype,
    index_col=index_col,
)
X_test = test_features.select_dtypes("number")


with open("./clipped_features.pkl", "rb") as f:
    clipped_features = pickle.load(f)
X_test = clipped_features.transform(X_test)
# アンサンブルのため統計値, nonscoredは入れない
# X_c = compute_row_statistics(X_test.loc[:, c_columns], prefix=c_prefix)
# X_g = compute_row_statistics(X_test.loc[:, g_columns], prefix=g_prefix)
# X_test = pd.concat([X_test, X_c, X_g], axis=1)


# lgbで予測
Y_test_pred = np.zeros((X_test.shape[0], len(columns)))
Y_test_pred = pd.DataFrame(Y_test_pred, columns=columns, index=test_features.index)
for target in columns:
    model_paths = glob.glob(f"./model_seed_*_{target}.jlb")
    for model_path in model_paths:
        model = joblib.load(model_path)
        Y_test_pred[target] += model.predict(X_test) / len(model_paths)
print(Y_test_pred.shape)
display(Y_test_pred)


# lgbの予測値補正
model_paths = glob.glob(f"./calibrate_model_target_*.jlb")
for model_path in model_paths:
    target = str(pathlib.Path(model_path).stem).replace("calibrate_model_target_", "")

    if target in columns:
        # print(target)
        model = joblib.load(model_path)
        X_targets = Y_test_pred.loc[:, target].values.reshape(-1, 1)
        Y_test_pred.loc[:, target] = model.predict_proba(X_targets)[:, 1]

print(Y_test_pred.shape)
display(Y_test_pred)
