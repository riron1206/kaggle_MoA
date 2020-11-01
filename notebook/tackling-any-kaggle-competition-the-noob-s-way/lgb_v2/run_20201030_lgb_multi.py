"""
sklearn.multioutputでlightGBMのモデル作成
- Feature-Engineering 追加
- MultilabelStratifiedKFold(n_splits=5, shuffle=False)
- データ全体を使用
- oof で計算
- feature engineering
- nonscored + ClassifierChain
- その他コード修正
Usage:
    $ conda activate py37

    # Powershellなら実行経過を画面表示とログファイルの両方に出力できる
    $ python run_20201030_lgb_multi.py -is_c -is_o_non | Tee-Object -FilePath ./run_20201030_lgb_multi/train.log
    $ python run_20201030_lgb_multi.py -m objective -is_c -is_o_non | Tee-Object -FilePath ./run_20201030_lgb_multi/objective.log

    $ poetry run python run_20201030_lgb_multi.py -is_c -is_o_non 2>&1

    $ python run_20201030_lgb_multi.py  # MultiOutputClassifierモデル作成
    $ python run_20201030_lgb_multi.py -is_c  # ClassifierChainモデル作成
    $ python run_20201030_lgb_multi.py -is_c -is_o_non  # nonscored + ClassifierChainモデル作成
    $ python run_20201030_lgb_multi.py -d  # デバッグ
    $ python run_20201030_lgb_multi.py -m objective  # パラメータチューニング
"""
import os
import sys
import glob
import datetime
import joblib
import random
import pathlib
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

# sys.path.append('../input/iterativestratification')
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
parser.add_argument(
    "-is_o_non",
    "--IS_ORDER_NONSCORED",
    action="store_const",
    const=True,
    default=False,
    help="Use ClassifierChain order train_targets_nonscored flag.",
)
args = vars(parser.parse_args())
# args = vars(parser.parse_args(args=[]))  # notebookで argparseそのままで実行する場合はこっち

# DEBUG = False
# DEBUG = True
# MODE = "train"
# MODE = "objective"
# IS_CHAIN = False
# IS_CHAIN = True
# IS_ORDER_NONSCORED = True
DEBUG = args["DEBUG"]
MODE = args["MODE"]
IS_CHAIN = args["IS_CHAIN"]
IS_ORDER_NONSCORED = args["IS_ORDER_NONSCORED"]
print("IS_CHAIN, IS_ORDER_NONSCORED:", IS_CHAIN, IS_ORDER_NONSCORED)

OUTDIR = r"run_20201030_lgb_multi"
os.makedirs(OUTDIR, exist_ok=True)

N_SPLITS = 5

N_TRIALS = 50

# SEEDS = [5, 12]
SEEDS = [0]  # 乱数シード指定してるが、StratifiedKFold は shuffle=False にしている

# lgb param
LR = 0.01
# N_ESTIMATORS = 1000
N_ESTIMATORS = 500

# DATADIR = '/kaggle/input/lish-moa/'
DATADIR = (
    r"C:\Users\81908\jupyter_notebook\poetry_work\tf23\01_MoA_compe\input\lish-moa"
)
train = pd.read_csv(f"{DATADIR}/train_features.csv")
test = pd.read_csv(f"{DATADIR}/test_features.csv")
train_targets = pd.read_csv(f"{DATADIR}/train_targets_scored.csv")
train_targets_nonscored = pd.read_csv(f"{DATADIR}/train_targets_nonscored.csv")
submission = pd.read_csv(f"{DATADIR}/sample_submission.csv")

# mean_log_loss計算用に確保
train_targets_orig = train_targets.copy()
train_targets_orig.drop(["sig_id"], inplace=True, axis=1)

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

    train_targets_orig = train_targets.copy()
    train_targets_orig.drop(["sig_id"], inplace=True, axis=1)

    # train_targets_nonscoredのクラスも減らす
    _classes = [
        "sig_id",
        "abc_transporter_expression_enhancer",
        "abl_inhibitor",
    ]
    train_targets_nonscored = train_targets_nonscored[_classes]

    N_TRIALS = 5
    LR = 0.1
    N_ESTIMATORS = 2


def mapping_and_filter(train, train_targets, test, train_targets_nonscored):
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
    train_targets_nonscored.drop(["sig_id"], inplace=True, axis=1)

    # nonscored と連結
    if IS_ORDER_NONSCORED:
        train_targets_nonscored = train_targets_nonscored.loc[
            :, ~(train_targets_nonscored.nunique() == 1)
        ]  # すべて0の列削除
        train_targets = pd.concat([train_targets_nonscored, train_targets], axis=1)

    print(train_targets.shape, train_targets_nonscored.shape)
    return train, train_targets, test, train_targets_nonscored


def load_model(model_path="model/fold00.model"):
    return joblib.load(model_path)


def get_features_gc(train, top_feat_cols=None):
    if top_feat_cols is None:
        top_feat_cols = train.columns
    features_g = [col for col in top_feat_cols if "g-" in col]
    features_c = [col for col in top_feat_cols if "c-" in col]
    return features_g, features_c


def get_features(train):
    """特徴量の列名取得"""
    features = train.columns.to_list()
    if "sig_id" in features:
        features.remove("sig_id")
    if "cp_type" in features:
        features.remove("cp_type")
    return features


# --------------------------------------- Feature-Engineering ---------------------------------------
def scaling(train, test, scaler=RobustScaler()):
    """規格化。pcaの後に実行してる。pcaの後だから外れ値にロバストな規格化使ってるみたい"""
    features = train.columns[2:]
    # scaler = RobustScaler()  # 外れ値に頑健な標準化
    scaler.fit(pd.concat([train[features], test[features]], axis=0))
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, get_features(train)


def fe_pca(
    train, test, features_g, features_c, n_components_g=70, n_components_c=10, SEED=123
):
    """pcaで圧縮した特徴量追加"""
    from sklearn.decomposition import PCA

    # 特徴量分けているが大区分がgとcの2区分あるので、それぞれでpca
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
    return train, test, get_features(train)


def fe_stats(
    train, test, features_g, features_c, params=["g", "c", "gc"], flag_add=False
):
    """統計量の特徴量追加"""
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
    return train, test, get_features(train)


def c_squared(train, test, features_c):
    """cの特徴量を2乗した特徴量作成"""
    for df in [train, test]:
        for feature in features_c:
            df[f"{feature}_squared"] = df[feature] ** 2
    return train, test, get_features(train)


def c_abs(train, test, features_c):
    """cの特徴量を絶対値とった特徴量作成"""
    for df in [train, test]:
        for feature in features_c:
            df[f"{feature}_abs"] = np.abs(df[feature])
    return train, test, get_features(train)


# ---------------------------------------------------------------------------------------------------


def save_model(model, model_path="model/fold00.model"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path, compress=True)


def mean_log_loss(y_true, y_pred, n_class=train_targets_orig.shape[1]):
    """マルチラベル全体でlog lossを平均する"""
    assert (
        y_true.shape[1] == n_class
    ), f"train_targetsの列数が {n_class} でないからlog_loss計算できない.y_true.shape: {y_true.shape}"

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    metrics = []
    for target in range(n_class):
        metrics.append(log_loss(y_true[:, target], y_pred[:, target]))
    return np.mean(metrics)


def run_multiout(params, seed):
    """MultiOutputClassifierでマルチラベル学習する"""
    # categorical_cols = ["cp_type", "cp_dose"]

    X_train = train.drop(["sig_id"], axis=1)
    X_test = test.drop(["sig_id"], axis=1)
    y_train = train_targets.copy()

    y_preds = []
    oof_pred = np.zeros([X_train.shape[0], train_targets_orig.shape[1]])

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
            if IS_ORDER_NONSCORED:
                random.seed(seed)

                n_t_non = train_targets_nonscored.shape[1]  # 332  # 402
                n_t_all = train_targets.shape[1]  # 206 + 332

                order = (
                    random.sample(range(n_t_non), k=n_t_non)
                    + random.sample(range(n_t_non, n_t_all), k=n_t_all - n_t_non)
                    if DEBUG == False
                    else list(range(n_t_non))
                    + random.sample(range(n_t_non, n_t_non + 2), k=2)
                )
                assert len(order) == y_train.shape[1]

                model = ClassifierChain(LGBMClassifier(**params), order=order)
            else:
                model = ClassifierChain(
                    LGBMClassifier(**params), order="random", random_state=seed
                )
            model_path = (
                f"{OUTDIR}/model/chain/fold{str(fold_id).zfill(2)}_{seed}.model"
            )
        else:
            model = MultiOutputClassifier(LGBMClassifier(**params))
            model_path = (
                f"{OUTDIR}/model/multi/fold{str(fold_id).zfill(2)}_{seed}.model"
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

        if IS_CHAIN == False:
            pred_y_val = np.array(pred_y_val)[:, :, 1].T  # take the positive class
            y_pred = np.array(y_pred)[:, :, 1].T  # take the positive class

        if IS_ORDER_NONSCORED:
            # targetの列だけにする
            pred_y_val = pred_y_val[:, train_targets_nonscored.shape[1] :]
            y_pred = y_pred[:, train_targets_nonscored.shape[1] :]

        oof_pred[valid_index] = pred_y_val
        y_preds.append(y_pred)

        if MODE == "train":
            save_model(
                model, model_path=model_path,
            )

    oof_score = mean_log_loss(train_targets_orig.values, oof_pred)
    print(f"oof_score: {oof_score}")

    return oof_pred, sum(y_preds) / len(y_preds)


def run_seed_avg(seeds=SEEDS):
    """シードアベレージ"""
    params = {
        # "num_leaves": 24,
        # "max_depth": 5,
        "num_leaves": 3,
        "max_depth": 2,
        "min_child_samples": 5243,
        "objective": "binary",
        "learning_rate": LR,
        "n_estimators": N_ESTIMATORS,
    }
    oofs = []
    subs = []
    for seed in seeds:
        print(f"\n================ seed:{seed} ================")
        _oof, _preds = run_multiout(params, seed)
        oofs.append(_oof)
        subs.append(_preds)
    oof_avg = sum(oofs) / len(seeds)
    sub_avg = sum(subs) / len(seeds)

    oof_score = mean_log_loss(train_targets_orig.values, oof_avg)
    oof_score_txt = (
        f"{OUTDIR}/oof_score_chain.txt" if IS_CHAIN else f"{OUTDIR}/oof_score_multi.txt"
    )
    with open(oof_score_txt, mode="w") as f:
        _str = f"oof_score seed_avg: {oof_score}"
        f.write(_str)
        print(_str)

    return oof_avg, sub_avg


def submit(test_pred, test, sample_submission):
    sample_submission.loc[:, train_targets_orig.columns] = test_pred
    sample_submission.loc[test["cp_type"] == 1, train_targets_orig.columns] = 0
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
    oof_score = mean_log_loss(train_targets_orig.values, oof)
    return np.mean(oof_score)


if __name__ == "__main__":
    features_g, features_c = get_features_gc(train)

    train, train_targets, test, train_targets_nonscored = mapping_and_filter(
        train, train_targets, test, train_targets_nonscored
    )

    train, test, features = fe_stats(
        train, test, features_g, features_c, flag_add=False
    )
    train, test, features = c_squared(train, test, features_c)
    train, test, features = c_abs(train, test, features_c)
    train, test, features = fe_pca(
        train,
        test,
        features_g,
        features_c,
        n_components_g=70,
        n_components_c=10,
        SEED=123,
    )
    train, test, features = scaling(train, test)

    print(
        f"### start:", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "###"
    )

    if MODE == "train":
        oof, sub = run_seed_avg()
        submit(sub, test, submission)
    else:
        _name = "_chain" if IS_CHAIN else "_multi"
        study = optuna.create_study(
            study_name=f"study{_name}",
            storage=f"sqlite:///{OUTDIR}/study{_name}.db",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=N_TRIALS)
        study.trials_dataframe().to_csv(
            f"{OUTDIR}/objective_history{_name}.csv", index=False
        )
        with open(f"{OUTDIR}/objective_best_params{_name}.txt", mode="w") as f:
            f.write(str(study.best_params))
        print(f"\nstudy.best_params:\n{study.best_params}")

    print(
        f"\n### end:", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "###"
    )


def model_load_submission():
    """model load + submission"""

    features_g, features_c = get_features_gc(train)

    train, train_targets, test, train_targets_nonscored = mapping_and_filter(
        train, train_targets, test, train_targets_nonscored
    )

    train, test, features = fe_stats(
        train, test, features_g, features_c, flag_add=False
    )
    train, test, features = c_squared(train, test, features_c)
    train, test, features = c_abs(train, test, features_c)
    train, test, features = fe_pca(
        train,
        test,
        features_g,
        features_c,
        n_components_g=70,
        n_components_c=10,
        SEED=123,
    )
    train, test, features = scaling(train, test)

    MODELDIR = f"{OUTDIR}/model"
    model_paths = (
        glob.glob(f"{MODELDIR}/chain/fold*")
        if IS_CHAIN
        else glob.glob(f"{MODELDIR}/multi/fold*")
    )

    X_test = test.drop(["sig_id"], axis=1)

    y_cols = (
        train_targets.iloc[:, train_targets_nonscored.shape[1] :].columns.to_list()
        if IS_ORDER_NONSCORED
        else train_targets.columns.to_list()
    )

    sub = pd.read_csv(f"{DATADIR}/sample_submission.csv")
    y_preds = []
    for m_path in tqdm(model_paths):
        model = load_model(m_path)

        y_pred = model.predict_proba(X_test)

        if IS_CHAIN == False:
            y_pred = np.array(y_pred)[:, :, 1].T  # take the positive class

        if IS_ORDER_NONSCORED:
            y_pred = y_pred[:, train_targets_nonscored.shape[1] :]

        y_preds.append(y_pred)
    _preds = sum(y_preds) / len(y_preds)
    _preds_df = pd.DataFrame(_preds, columns=y_cols)
    for col in y_cols:
        sub[col] = _preds_df[col]

    # Postprocessing: cp_typeが'ctl_vehicle'の行は予測値を0に設定
    sub.loc[test["cp_type"] == 1, y_cols] = 0
    sub.to_csv("submission.csv", index=False)

    print(sub.shape)
    sub
