"""
クラスごとにlightGBMのモデル作成
Usage:
    $ conda activate py37
    $ python 20201021_lgb_train.py
"""
import os
import sys
import glob
import pathlib
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")

DATADIR = r"C:\Users\yokoi.shingo\my_task\MoA_Prediction\input\lish-moa"

OUTDIR = r"20201021_lgb_train"
# OUTDIR = r"tmp"
os.makedirs(OUTDIR, exist_ok=True)

DEBUG = False
# DEBUG = True

# MODE = "train"
MODE = "objective"

# 5foldにするとkaggle data setにupできない。1000ファイルまでしか上げれないみたいなので
N_SPLITS = 4

train = pd.read_csv(f"{DATADIR}/train_features.csv")
test = pd.read_csv(f"{DATADIR}/test_features.csv")
train_targets_scored = pd.read_csv(f"{DATADIR}/train_targets_scored.csv")
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
    train_targets_scored = train_targets_scored[_classes]
    submission = submission[_classes]


def preprocess(df):
    df = df.copy()
    # カテゴリ型のラベルを2値化
    df.loc[:, "cp_type"] = df.loc[:, "cp_type"].map({"trt_cp": 0, "ctl_vehicle": 1})
    df.loc[:, "cp_dose"] = df.loc[:, "cp_dose"].map({"D1": 0, "D2": 1})
    return df


def save_model(model, model_path="model/fold00.model"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path, compress=True)


def load_model(model_path="model/fold00.model"):
    return joblib.load(model_path)


def run_lgb(target_col: str, cv, params):

    categorical_cols = ["cp_type", "cp_dose"]

    X_train = train.drop(["sig_id"], axis=1)
    y_train = train_targets_scored[target_col]
    X_test = test.drop(["sig_id"], axis=1)

    y_preds = []
    # models = []
    oof_train = np.zeros((len(X_train),))

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
        print(f"\n------------ fold:{fold_id} ------------")
        X_tr = X_train.loc[train_index, :]
        X_val = X_train.loc[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)

        lgb_eval = lgb.Dataset(
            X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols
        )

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            verbose_eval=300,
            num_boost_round=1000,
            # num_boost_round=100,
            early_stopping_rounds=300,
        )

        oof_train[valid_index] = model.predict(
            X_val, num_iteration=model.best_iteration
        )
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        y_preds.append(y_pred)

        # models.append(model)
        if MODE == "train":
            save_model(
                model,
                model_path=f"{OUTDIR}/model/{target_col}/fold{str(fold_id).zfill(2)}.model",
            )

    return oof_train, sum(y_preds) / len(y_preds)


def run_cv(params, seeds=[5]):
    """乱数変えてcv実行"""
    _cols = train_targets_scored.columns.to_list()[:]
    _cols.remove("sig_id")

    oof = train_targets_scored.copy()
    sub = submission.copy()
    oof.loc[:, _cols] = 0.0
    sub.loc[:, _cols] = 0.0
    # print(oof)

    for seed in seeds:
        print(f"\n================ seed:{seed} ================")
        # cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
        cv = KFold(n_splits=N_SPLITS, shuffle=False)

        for i, target_col in enumerate(train_targets_scored.columns):
            if target_col != "sig_id":
                print(f"\n########## {i} target_col:{target_col} ##########")
                _oof, _preds = run_lgb(target_col, cv, params)
                oof[target_col] += _oof
                sub[target_col] += _preds
    oof.loc[:, _cols] /= len(seeds)
    sub.loc[:, _cols] /= len(seeds)

    return oof, sub


def objective(trial):
    params = {
        "objective": "binary",
        "learning_rate": 0.1,
    }
    params["max_depth"] = trial.suggest_int("max_depth", 1, 7)
    params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2 ** params["max_depth"])
    params["min_data_in_leaf"] = trial.suggest_int(
        "min_data_in_leaf",
        1,
        max(
            1, int(train.shape[0] * ((N_SPLITS - 1) / N_SPLITS) / params["num_leaves"])
        ),
    )
    oof, sub = run_cv(params)

    scores = []
    for target_col in train_targets_scored.columns:
        if target_col != "sig_id":
            scores.append(log_loss(train_targets_scored[target_col], oof[target_col]))

    return np.mean(scores)


def main_train():
    params = {
        "num_leaves": 24,
        "max_depth": 5,
        "objective": "binary",
        "learning_rate": 0.01,
    }
    oof, sub = run_cv(params)

    scores = []
    for target_col in train_targets_scored.columns:
        if target_col != "sig_id":
            scores.append(log_loss(train_targets_scored[target_col], oof[target_col]))
    oof_metrics = f"oof_log_loss:, {np.mean(scores)}"
    with open(f"{OUTDIR}/oof_metrics.txt", mode="w") as f:
        f.write(oof_metrics)
    print(oof_metrics)

    # Postprocessing: cp_typeが'ctl_vehicle'の行は予測値を0に設定
    _cols = train_targets_scored.columns.to_list()[:]
    _cols.remove("sig_id")
    sub.loc[test["cp_type"] == 1, _cols] = 0
    sub.to_csv(f"{OUTDIR}/submission.csv", index=False)


if __name__ == "__main__":
    # Preprocessing
    train = preprocess(train)
    test = preprocess(test)

    if MODE == "train":
        main_train()
    else:
        study = optuna.create_study()
        study.optimize(objective, n_trials=50)
        study.trials_dataframe().to_csv(f"{OUTDIR}/objective_history.csv", index=False)
        with open(f"{OUTDIR}/objective_best_params.txt", mode="w") as f:
            f.write(str(study.best_params))
        print(f"\nstudy.best_params:\n{study.best_params}")
