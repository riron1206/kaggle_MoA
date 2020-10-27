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

# +
# Usage:
# poetry run python 20201026_lgb_multi.py
import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
import optuna
from tqdm import tqdm

sys.path.append(r"C:\Users\81908\Git\iterative-stratification")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

warnings.filterwarnings("ignore")

OUTDIR = r"20201026_lgb_multi"
os.makedirs(OUTDIR, exist_ok=True)

DEBUG = False
# DEBUG = True

MODE = "train"
# MODE = "objective"

N_SPLITS = 5

# DATADIR = '/kaggle/input/lish-moa/'
DATADIR = (
    r"C:\Users\81908\jupyter_notebook\poetry_work\tf23\01_MoA_compe\input\lish-moa"
)
train = pd.read_csv(f"{DATADIR}/train_features.csv")
test = pd.read_csv(f"{DATADIR}/test_features.csv")
train_targets = pd.read_csv(f"{DATADIR}/train_targets_scored.csv")
train_targets_nonscored = pd.read_csv(f"{DATADIR}/train_targets_nonscored.csv")
submission = pd.read_csv(f"{DATADIR}/sample_submission.csv")

if DEBUG:
    np.random.seed(0)  # 乱数シード固定
    #    # ランダムに2000件選択
    #    _ids = np.random.choice(train.index, 2000)
    #    train = train.loc[_ids].reset_index(drop=True)
    #    train_targets_scored = train_targets_scored.loc[_ids].reset_index(drop=True)

    # 3クラスのみにする
    _classes = [
        "sig_id",
        "5-alpha_reductase_inhibitor",
        "11-beta-hsd1_inhibitor",
        # "acat_inhibitor", # 2000件だとすべて0になるのでダメ
    ]
    train_targets = train_targets[_classes]
    submission = submission[_classes]


# -


def mapping_and_filter(train, train_targets, test):
    """前処理"""
    cp_type = {"trt_cp": 0, "ctl_vehicle": 1}
    cp_dose = {"D1": 0, "D2": 1}
    for df in [train, test]:
        df["cp_type"] = df["cp_type"].map(cp_type)
        df["cp_dose"] = df["cp_dose"].map(cp_dose)
    # ctl_vehicleは必ず0なので学習データから除く
    train_targets = train_targets[train["cp_type"] == 0].reset_index(drop=True)
    train = train[train["cp_type"] == 0].reset_index(drop=True)
    # sig_id列はidなので不要
    train_targets.drop(["sig_id"], inplace=True, axis=1)
    return train, train_targets, test


# +
def save_model(model, model_path="model/fold00.model"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path, compress=True)


def load_model(model_path="model/fold00.model"):
    return joblib.load(model_path)


# -


def mean_log_loss(y_true, y_pred):
    """マルチラベル全体でlog lossを平均する"""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    metrics = []
    for target in range(y_true.shape[1]):
        metrics.append(log_loss(y_true[:, target], y_pred[:, target]))
    return np.mean(metrics)


# + code_folding=[]
def run_multiout(model, seed):
    """MultiOutputClassifierでマルチラベル学習する"""
    # categorical_cols = ["cp_type", "cp_dose"]

    X_train = train.drop(["sig_id"], axis=1)
    y_train = train_targets.copy()
    X_test = test.drop(["sig_id"], axis=1)

    y_preds = []
    oof_pred = np.zeros([X_train.shape[0], y_train.shape[1]])

    # for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
    for fold_id, (train_index, valid_index) in tqdm(
        enumerate(
            MultilabelStratifiedKFold(
                n_splits=N_SPLITS, random_state=seed, shuffle=True
            ).split(y_train, y_train)
        )
    ):
        X_tr, X_val = (
            X_train.values[train_index],
            X_train.values[valid_index],
        )
        y_tr, y_val = (
            y_train.values[train_index],
            y_train.values[valid_index],
        )

        # MultiOutputClassifier はval使えないみたい
        # https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html
        model.fit(
            X_tr,
            y_tr,
            # categorical_feature=categorical_cols  # MultiOutputClassifier では指定できない
            # eval_metric="error",
            # verbose=300,
            # eval_set=[(X_tr, y_tr), (X_val, y_val)],
            # early_stopping_rounds=300,
        )

        pred_y_val = model.predict_proba(X_val)
        pred_y_val = np.array(pred_y_val)[:, :, 1].T  # take the positive class
        oof_pred[valid_index] = pred_y_val

        y_pred = model.predict_proba(X_test)
        y_pred = np.array(y_pred)[:, :, 1].T  # take the positive class
        y_preds.append(y_pred)

        if MODE == "train":
            save_model(
                model, model_path=f"{OUTDIR}/model/fold{str(fold_id).zfill(2)}.model",
            )

    oof_score = mean_log_loss(train_targets.values, oof_pred)
    print(f"oof_score: {oof_score}")

    return oof_pred, sum(y_preds) / len(y_preds)


# -


def run_seed_avg(model, seeds=[5, 12]):
    """シードアベレージ"""
    oofs = []
    subs = []
    for seed in seeds:
        print(f"\n================ seed:{seed} ================")
        _oof, _preds = run_multiout(model, seed)
        oofs.append(_oof)
        subs.append(_preds)
    oof_avg = sum(oofs) / len(seeds)
    sub_avg = sum(subs) / len(seeds)

    oof_score = mean_log_loss(train_targets.values, oof_avg)
    print(f"oof_score seed_avg: {oof_score}")

    return oof_avg, sub_avg


def submit(test_pred, test, sample_submission, train_targets):
    sample_submission.loc[:, train_targets.columns] = test_pred
    sample_submission.loc[test["cp_type"] == 1, train_targets.columns] = 0
    sample_submission.to_csv(f"{OUTDIR}/submission.csv", index=False)
    return sample_submission


def objective(trial):
    params = {
        "objective": "binary",
        "learning_rate": 0.1,
    }
    params["max_depth"] = trial.suggest_int("max_depth", 1, 7)
    params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2 ** params["max_depth"])
    params["min_child_samples"] = trial.suggest_int(
        "min_child_samples",
        1,
        max(
            1, int(train.shape[0] * ((N_SPLITS - 1) / N_SPLITS) / params["num_leaves"])
        ),
    )
    model = MultiOutputClassifier(LGBMClassifier(**params))
    oof, sub = run_multiout(model, 42)
    oof_score = mean_log_loss(train_targets.values, oof)
    return np.mean(oof_score)


def main_train():
    params = {
        "num_leaves": 24,
        "max_depth": 5,
        "objective": "binary",
        "learning_rate": 0.01,
        "n_estimators": 1000,
    }
    model = MultiOutputClassifier(LGBMClassifier(**params))
    oof, sub = run_seed_avg(model)
    submit(sub, test, submission, train_targets)


if __name__ == "__main__":
    train, train_targets, test = mapping_and_filter(train, train_targets, test)

    if MODE == "train":
        main_train()
    else:
        study = optuna.create_study(
            study_name="study",
            storage=f"sqlite:///{OUTDIR}/study.db",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=5)
        study.trials_dataframe().to_csv(f"{OUTDIR}/objective_history.csv", index=False)
        with open(f"{OUTDIR}/objective_best_params.txt", mode="w") as f:
            f.write(str(study.best_params))
        print(f"\nstudy.best_params:\n{study.best_params}")
