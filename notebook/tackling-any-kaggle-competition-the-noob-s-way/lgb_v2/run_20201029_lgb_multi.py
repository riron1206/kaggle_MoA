"""
sklearn.multioutputでlightGBMのモデル作成
- Feature-Engineering 追加
- MultilabelStratifiedKFold(n_splits=5, shuffle=False)
- データ全体を使用
- oof で計算
- その他コード修正
Usage:
    $ conda activate py37

    # Powershellなら実行経過を画面表示とログファイルの両方に出力できる
    $ python run_20201029_lgb_multi.py | Tee-Object -FilePath ./run_20201029_lgb_multi/train.log
    $ python run_20201029_lgb_multi.py -m objective | Tee-Object -FilePath ./run_20201029_lgb_multi/objective.log

    $ poetry run python run_20201029_lgb_multi.py -is_c 2>&1

    $ python run_20201029_lgb_multi.py -is_c  # ClassifierChainモデル作成
    $ python run_20201029_lgb_multi.py -d  # デバッグ
    $ python run_20201029_lgb_multi.py -m objective  # パラメータチューニング
"""
import os
import sys
import datetime
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
import optuna
from tqdm import tqdm
import argparse

sys.path.append(r"C:\Users\81908\Git\iterative-stratification")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

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
parser.add_argument(
    "-is_c",
    "--IS_CHAIN",
    action="store_const",
    const=True,
    default=False,
    help="ClassifierChain flag.",
)
args = vars(parser.parse_args())
# args = vars(parser.parse_args(args=[])) # notebookで argparseそのままで実行する場合はこっち

# DEBUG = False
# DEBUG = True
# MODE = "train"
# MODE = "objective"
# IS_CHAIN = False
# IS_CHAIN = True
DEBUG = args["DEBUG"]
MODE = args["MODE"]
IS_CHAIN = args["IS_CHAIN"]


OUTDIR = r"run_20201029_lgb_multi"
os.makedirs(OUTDIR, exist_ok=True)

N_SPLITS = 5

N_TRIALS = 50

# SEEDS = [5, 12]
SEEDS = [0]  # 乱数シード指定してるが、StratifiedKFold は shuffle=False にしている

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

    N_TRIALS = 5


def mapping_and_filter(train, train_targets, test):
    """前処理"""
    cp_type = {"trt_cp": 0, "ctl_vehicle": 1}
    cp_dose = {"D1": 0, "D2": 1}
    for df in [train, test]:
        df["cp_type"] = df["cp_type"].map(cp_type)
        df["cp_dose"] = df["cp_dose"].map(cp_dose)

    ## ctl_vehicleは必ず0なので学習データから除く
    # train_targets = train_targets[train["cp_type"] == 0].reset_index(drop=True)
    # train = train[train["cp_type"] == 0].reset_index(drop=True)

    # sig_id列はidなので不要
    train_targets.drop(["sig_id"], inplace=True, axis=1)
    return train, train_targets, test


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

    features = train.columns[2:]
    return train, test, features


def fe_stats(train, test, params=["g", "c", "gc"], flag_add=False):
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

    features = train.columns[2:]
    return train, test, features


def c_squared(train, test):
    """cの特徴量を2乗した特徴量作成"""
    features_c = list(train.columns[776:876])
    for df in [train, test]:
        for feature in features_c:
            df[f"{feature}_squared"] = df[feature] ** 2
    features = train.columns[2:]
    return train, test, features


def c_abs(train, test):
    """cの特徴量を絶対値とった特徴量作成"""
    features_c = list(train.columns[776:876])
    for df in [train, test]:
        for feature in features_c:
            df[f"{feature}_abs"] = np.abs(df[feature])
    features = train.columns[2:]
    return train, test, features


# ---------------------------------------------------------------------------------------------------


def save_model(model, model_path="model/fold00.model"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path, compress=True)


def load_model(model_path="model/fold00.model"):
    return joblib.load(model_path)


def mean_log_loss(y_true, y_pred):
    """マルチラベル全体でlog lossを平均する"""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    metrics = []
    for target in range(y_true.shape[1]):
        metrics.append(log_loss(y_true[:, target], y_pred[:, target]))
    return np.mean(metrics)


def run_multiout(params, seed):
    """MultiOutputClassifierでマルチラベル学習する"""
    # categorical_cols = ["cp_type", "cp_dose"]

    X_train = train.drop(["sig_id"], axis=1)
    y_train = train_targets.copy()
    X_test = test.drop(["sig_id"], axis=1)

    y_preds = []
    oof_pred = np.zeros([X_train.shape[0], y_train.shape[1]])

    ## for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
    # for fold_id, (train_index, valid_index) in tqdm(
    #    enumerate(
    #        MultilabelStratifiedKFold(
    #            n_splits=N_SPLITS, random_state=seed, shuffle=True
    #        ).split(y_train, y_train)
    #    )
    # ):
    # MultiLabelStratifiedKFold(n_splits=5, shuffle=False) で乱数固定する 20201028
    for fold_id, (train_index, valid_index) in tqdm(
        enumerate(
            MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=False).split(
                y_train, y_train
            )
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

        if IS_CHAIN:
            model = ClassifierChain(LGBMClassifier(**params), random_state=seed)
            model_path = (
                f"{OUTDIR}/model/chain_fold{str(fold_id).zfill(2)}_{seed}.model"
            )
        else:
            model = MultiOutputClassifier(LGBMClassifier(**params))
            model_path = (
                f"{OUTDIR}/model/multi_fold{str(fold_id).zfill(2)}_{seed}.model"
            )

        # MultiOutputClassifier/ClassifierChain はval使えないみたい
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
        y_pred = model.predict_proba(X_test)
        # print(pred_y_val, np.array(pred_y_val).shape)

        if IS_CHAIN == False:
            pred_y_val = np.array(pred_y_val)[:, :, 1].T  # take the positive class
            y_pred = np.array(y_pred)[:, :, 1].T  # take the positive class
        # print(y_pred.shape)

        oof_pred[valid_index] = pred_y_val
        y_preds.append(y_pred)

        if MODE == "train":
            save_model(
                model, model_path=model_path,
            )

    oof_score = mean_log_loss(train_targets.values, oof_pred)
    print(f"oof_score: {oof_score}")

    return oof_pred, sum(y_preds) / len(y_preds)


def run_seed_avg(params, seeds=SEEDS):
    """シードアベレージ"""
    oofs = []
    subs = []
    for seed in seeds:
        print(f"\n================ seed:{seed} ================")
        _oof, _preds = run_multiout(params, seed)
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
    oof, sub = run_multiout(params, SEEDS[0])
    oof_score = mean_log_loss(train_targets.values, oof)
    return np.mean(oof_score)


def main_train():
    params = {
        "num_leaves": 24,
        "max_depth": 5,
        "objective": "binary",
        "learning_rate": 0.01,
        "n_estimators": 3000,
        # "n_estimators": 100,
    }
    oof, sub = run_seed_avg(params)

    if IS_CHAIN:
        out_metric_txt = f"{OUTDIR}/oof_metric_chain.txt"
    else:
        out_metric_txt = f"{OUTDIR}/oof_metric_multi.txt"
    oof_metric = f"oof_log_loss:, {mean_log_loss(train_targets, oof)}"
    with open(out_metric_txt, mode="w") as f:
        f.write(oof_metric)
    print(oof_metric)

    submit(sub, test, submission, train_targets)


if __name__ == "__main__":
    print(
        f"### start:", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "###"
    )

    train, train_targets, test = mapping_and_filter(train, train_targets, test)
    train, test, features = fe_stats(train, test, flag_add=False)
    train, test, features = c_squared(train, test)
    train, test, features = c_abs(train, test)
    train, test, features = fe_pca(
        train, test, n_components_g=70, n_components_c=10, SEED=123
    )
    train, test, features = scaling(train, test)

    if MODE == "train":
        main_train()
    else:
        if IS_CHAIN:
            _name = "_chain"
        else:
            _name = "_multi"

        study = optuna.create_study(
            study_name=f"study{_name}",
            storage=f"sqlite:///{OUTDIR}/study{_name}.db",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=N_TRIALS)
        study.trials_dataframe().to_csv(
            f"{OUTDIR}/objective_history{_name}.csv", index=False
        )
        with open(f"{OUTDIR}/objective_best_params.txt", mode="w") as f:
            f.write(str(study.best_params))
        print(f"\nstudy.best_params:\n{study.best_params}")

    print(
        f"### end:", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "###"
    )
