"""
クラスごとにlightGBMのモデル作成
- Feature-Engineering 追加
- StratifiedKFold 使用
- StratifiedKFold(n_splits=5, shuffle=False)
- データ全体を使用
- oof で計算
- その他コード修正
Usage:
    $ conda activate py37

    # Powershellなら実行経過を画面表示とログファイルの両方に出力できる
    $ python 20201028_lgb_train.py | Tee-Object -FilePath ./20201028_lgb_train/train.log
    $ python 20201028_lgb_train.py -m objective | Tee-Object -FilePath ./20201028_lgb_train/objective.log

    $ python 20201028_lgb_train.py -d  # デバッグ
    $ python 20201028_lgb_train.py -m objective  # パラメータチューニング
"""
import datetime
import logging
import os
import sys
import glob
import pathlib
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
import optuna
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--DEBUG",
    action="store_const",
    const=True,
    default=False,
    help="DEBUG flag.",
)
parser.add_argument(
    "-m", "--MODE", type=str, default="train", help="MODE flag.",
)
args = vars(parser.parse_args())

# DEBUG = False
# DEBUG = True
# MODE = "train"
# MODE = "objective"
DEBUG = args["DEBUG"]
MODE = args["MODE"]

OUTDIR = r"20201028_lgb_train"
os.makedirs(OUTDIR, exist_ok=True)

N_SPLITS = 5

# SEEDS = [5, 12]
SEEDS = [0]  # 乱数シード指定してるが、StratifiedKFold は shuffle=False にしている

N_TRIALS = 50

# DATADIR = '/kaggle/input/lish-moa/'
DATADIR = r"C:\Users\yokoi.shingo\my_task\MoA_Prediction\input\lish-moa"
train = pd.read_csv(f"{DATADIR}/train_features.csv")
test = pd.read_csv(f"{DATADIR}/test_features.csv")
train_targets = pd.read_csv(f"{DATADIR}/train_targets_scored.csv")
# train_targets_nonscored = pd.read_csv(f"{DATADIR}/train_targets_nonscored.csv")
submission = pd.read_csv(f"{DATADIR}/sample_submission.csv")

if DEBUG:
    np.random.seed(0)  # 乱数シード固定
    #    # ランダムに2000件選択
    #    _ids = np.random.choice(train.index, 2000)
    #    train = train.loc[_ids].reset_index(drop=True)
    #    train_targets = train_targets.loc[_ids].reset_index(drop=True)

    # 3クラスのみにする
    _classes = [
        "sig_id",
        "5-alpha_reductase_inhibitor",
        "11-beta-hsd1_inhibitor",
        # "acat_inhibitor", # 2000件だとすべて0になるのでダメ
    ]
    train_targets = train_targets[_classes]
    submission = submission[_classes]

    SEEDS = [0]
    # SEEDS = [5, 12]
    N_SPLITS = 2
    N_TRIALS = 2


def mapping_and_filter(train, train_targets, test):
    """前処理"""
    cp_type = {"trt_cp": 0, "ctl_vehicle": 1}
    cp_dose = {"D1": 0, "D2": 1}
    for df in [train, test]:
        df["cp_type"] = df["cp_type"].map(cp_type)
        df["cp_dose"] = df["cp_dose"].map(cp_dose)

    # 全データでoofだす 20201028
    ## ctl_vehicleは必ず0なので学習データから除く
    # train_targets = train_targets[train["cp_type"] == 0].reset_index(drop=True)
    # train = train[train["cp_type"] == 0].reset_index(drop=True)

    # sig_id列はidなので不要
    train_targets.drop(["sig_id"], inplace=True, axis=1)

    print(f"train.shape: {train.shape}")
    return train, train_targets, test


def mean_log_loss(train_targets, oof):
    """マルチラベル全体でlog lossを平均する"""
    scores = []
    for target_col in train_targets.columns:
        # oof にはEvaluation に書いてるのと同じクリップをしてからlog_loss 出す
        oof_clip = np.clip(oof[target_col], 1e-15, 1 - 1e-15)
        scores.append(log_loss(train_targets[target_col], oof_clip))
    return np.mean(scores)


def submit(test_pred, test, sample_submission, train_targets, s_csv="submission.csv"):
    sample_submission.loc[:, train_targets.columns] = test_pred
    sample_submission.loc[test["cp_type"] == 1, train_targets.columns] = 0
    sample_submission.to_csv(s_csv, index=False)
    return sample_submission


# --------------------------------------- Feature-Engineering ---------------------------------------
def scaling(train, test, scaler=RobustScaler()):
    """規格化。pcaの後に実行してる。pcaの後だから外れ値にロバストな規格化使ってるみたい"""
    features = train.columns[2:]
    # scaler = RobustScaler()  # 外れ値に頑健な標準化
    scaler.fit(pd.concat([train[features], test[features]], axis=0))
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, features


def fe_pca(train, test, n_components_g=70, n_components_c=10, SEED=123):
    """pcaで圧縮した特徴量追加"""

    # 特徴量分けているが大区分がgとcの2区分あるので、それぞれでpca
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])

    def create_pca(train, test, features, kind="g", n_components=n_components_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis=0)
        pca = PCA(n_components=n_components, random_state=SEED)
        data = pca.fit_transform(data)
        columns = [f"pca_{kind}{i + 1}" for i in range(n_components)]
        data = pd.DataFrame(data, columns=columns)
        train_ = data.iloc[: train.shape[0]]
        test_ = data.iloc[train.shape[0] :].reset_index(drop=True)
        train = pd.concat([train, train_], axis=1)
        test = pd.concat([test, test_], axis=1)
        return train, test

    train, test = create_pca(
        train, test, features_g, kind="g", n_components=n_components_g
    )
    train, test = create_pca(
        train, test, features_c, kind="c", n_components=n_components_c
    )
    return train, test


def fe_stats(train, test, params=["g", "c", "gc"], flag_add=True):
    """統計量の特徴量追加"""

    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])

    for df in [train, test]:
        if "g" in params:
            df["g_sum"] = df[features_g].sum(axis=1)
            df["g_mean"] = df[features_g].mean(axis=1)
            df["g_std"] = df[features_g].std(axis=1)
            df["g_kurt"] = df[features_g].kurtosis(axis=1)
            df["g_skew"] = df[features_g].skew(axis=1)
            if flag_add:
                df["g_quan25"] = df[features_g].quantile(0.25, axis=1)
                df["g_quan75"] = df[features_g].quantile(0.75, axis=1)
                # df["g_quan_ratio"] = df["g_quan75"] / df["g_quan25"]
                df["g_ptp"] = np.abs(df[features_g].max(axis=1)) - np.abs(
                    df[features_g].min(axis=1)
                )
        if "c" in params:
            df["c_sum"] = df[features_c].sum(axis=1)
            df["c_mean"] = df[features_c].mean(axis=1)
            df["c_std"] = df[features_c].std(axis=1)
            df["c_kurt"] = df[features_c].kurtosis(axis=1)
            df["c_skew"] = df[features_c].skew(axis=1)
            if flag_add:
                df["c_quan25"] = df[features_c].quantile(0.25, axis=1)
                df["c_quan75"] = df[features_c].quantile(0.75, axis=1)
                # df["c_quan_ratio"] = df["c_quan75"] / df["c_quan25"]
                df["c_ptp"] = np.abs(df[features_c].max(axis=1)) - np.abs(
                    df[features_c].min(axis=1)
                )
        if "gc" in params:
            df["gc_sum"] = df[features_g + features_c].sum(axis=1)
            df["gc_mean"] = df[features_g + features_c].mean(axis=1)
            df["gc_std"] = df[features_g + features_c].std(axis=1)
            df["gc_kurt"] = df[features_g + features_c].kurtosis(axis=1)
            df["gc_skew"] = df[features_g + features_c].skew(axis=1)
            if flag_add:
                df["gc_quan25"] = df[features_g + features_c].quantile(0.25, axis=1)
                df["gc_quan75"] = df[features_g + features_c].quantile(0.75, axis=1)
                # df["gc_quan_ratio"] = df["gc_quan75"] / df["gc_quan25"]
                df["gc_ptp"] = np.abs(df[features_g + features_c].max(axis=1)) - np.abs(
                    df[features_g + features_c].min(axis=1)
                )

    return train, test


def c_squared(train, test):
    """cの特徴量について2乗した特徴量作成"""
    features_c = list(train.columns[776:876])
    for df in [train, test]:
        for feature in features_c:
            df[f"{feature}_squared"] = df[feature] ** 2
    return train, test


# ---------------------------------------------------------------------------------------------------


def save_model(model, model_path="model/fold00.model"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path, compress=True)


def load_model(model_path="model/fold00.model"):
    return joblib.load(model_path)


def run_lgb(target_col: str, params, seed):
    """1クラスだけ学習"""

    categorical_cols = ["cp_type", "cp_dose"]

    X_train = train.drop(["sig_id"], axis=1)
    y_train = train_targets[target_col]
    X_test = test.drop(["sig_id"], axis=1)

    y_preds = []
    f_importances = []
    oof_train = np.zeros((len(X_train),))

    # cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    # cv = KFold(n_splits=N_SPLITS, shuffle=False)
    # for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):

    # cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)  # シャッフルしない 20201028
    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
        print(f"\n------------ fold:{fold_id} ------------")
        X_tr = X_train.loc[train_index, :]
        X_val = X_train.loc[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)
        lgb_eval = lgb.Dataset(
            X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols
        )

        if MODE == "train":
            num_boost_round = 10000
        else:
            num_boost_round = 1000

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            verbose_eval=300,
            num_boost_round=num_boost_round,
            early_stopping_rounds=300,
        )

        oof_train[valid_index] = model.predict(
            X_val, num_iteration=model.best_iteration
        )

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_preds.append(y_pred)

        if MODE == "train":
            save_model(
                model,
                model_path=f"{OUTDIR}/model/{target_col}/fold{str(fold_id).zfill(2)}_{seed}.model",
            )
            f_importance = np.array(model.feature_importance(importance_type="gain"))
            f_importances.append(f_importance)

    if MODE == "train":
        f_importances_avg = sum(f_importances) / len(f_importances)
        importance_df = pd.DataFrame(
            {"feature": model.feature_name(), "importance": f_importances_avg}
        )
    else:
        importance_df = None

    return (
        oof_train,
        sum(y_preds) / len(y_preds),
        importance_df,
    )


def run_all_target(params, seeds=SEEDS):
    """乱数変えて全クラス学習"""
    _cols = train_targets.columns.to_list()[:]

    oof = train_targets.copy()
    sub = submission.copy()
    oof.loc[:, _cols] = 0.0
    sub.loc[:, _cols] = 0.0
    importance_dfs = []

    for seed in seeds:
        print(f"\n================ seed:{seed} ================")
        for i, target_col in enumerate(train_targets.columns):
            print(f"\n########## {i} target_col:{target_col} ##########")
            _oof, _preds, _importance_df = run_lgb(target_col, params, seed)
            oof[target_col] += _oof
            sub[target_col] += _preds
            importance_dfs.append(_importance_df)

    # seed_avg
    oof.loc[:, _cols] /= len(seeds)
    sub.loc[:, _cols] /= len(seeds)

    # feature_importance plot
    if MODE == "train":
        _imp = None
        for _df in importance_dfs:
            if _imp is None:
                _imp = _df["importance"].values
            else:
                _imp += _df["importance"].values
        f_importances_avg = _imp / len(importance_dfs)
        importance_df = pd.DataFrame(
            {
                "feature": importance_dfs[0]["feature"].to_list(),
                "importance": f_importances_avg,
            }
        )
        display_importances(importance_df,)

    return oof, sub


def display_importances(
    importance_df, png_path=f"{OUTDIR}/feature_importance.png",
):
    """feature_importance plot"""
    importance_df.sort_values(by="importance", ascending=False).to_csv(
        f"{OUTDIR}/feature_importance.csv"
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
    oof, _ = run_all_target(params, seeds=[SEEDS[0]])
    return mean_log_loss(train_targets, oof)


def main_train():
    params = {
        "num_leaves": 2,
        "max_depth": 1,
        "min_data_in_leaf": 969,
        "objective": "binary",
        "learning_rate": 0.01,
    }
    oof, sub = run_all_target(params)

    oof_metric = f"oof_log_loss:, {mean_log_loss(train_targets, oof)}"
    with open(f"{OUTDIR}/oof_metric.txt", mode="w") as f:
        f.write(oof_metric)
    print(oof_metric)

    submit(sub, test, submission, train_targets, s_csv=f"{OUTDIR}/submission.csv")


if __name__ == "__main__":
    print(
        f"### start:", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "###"
    )

    train, train_targets, test = mapping_and_filter(train, train_targets, test)
    train, test = fe_stats(train, test)
    train, test = c_squared(train, test)
    train, test = fe_pca(train, test, n_components_g=70, n_components_c=10, SEED=123)
    train, test, features = scaling(train, test)

    if MODE == "train":
        main_train()
    else:
        study = optuna.create_study(
            study_name="study",
            storage=f"sqlite:///{OUTDIR}/study.db",
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=1),
        )
        study.optimize(objective, n_trials=N_TRIALS)
        study.trials_dataframe().to_csv(f"{OUTDIR}/objective_history.csv", index=False)
        with open(f"{OUTDIR}/objective_best_params.txt", mode="w") as f:
            f.write(str(study.best_params))
        print(f"\nstudy.best_params:\n{study.best_params}")

    print(
        f"### end:", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "###"
    )
