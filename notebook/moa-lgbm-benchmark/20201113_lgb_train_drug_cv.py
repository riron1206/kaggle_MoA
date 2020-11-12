"""
クラスごとにlightGBMのモデル作成
- Feature-Engineering 追加
- 薬idでグルーピングしたlGroupStratifiedKFold
- データ全体を使用
- oof で計算
- その他コード修正
Usage:
    $ conda activate py37

    # Powershellなら実行経過を画面表示とログファイルの両方に出力できる
    $ python 20201113_lgb_train_drug_cv.py | Tee-Object -FilePath ./20201113_lgb_train_drug_cv/train.log
    $ python 20201113_lgb_train_drug_cv.py -m objective | Tee-Object -FilePath ./20201113_lgb_train_drug_cv/objective.log

    $ poetry run python 20201113_lgb_train_drug_cv.py -d

    $ python 20201113_lgb_train_drug_cv.py -d  # デバッグ
    $ python 20201113_lgb_train_drug_cv_drug_cv.py -m objective  # パラメータチューニング
"""
import re
import os
import sys
import pickle
import joblib
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler
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
    "-m", "--MODE", type=str, default="train", help="MODE flag.",  # "objective"
)
args = vars(parser.parse_args())
DEBUG = args["DEBUG"]
MODE = args["MODE"]

OUTDIR = r"20201113_lgb_train_drug_cv"
os.makedirs(OUTDIR, exist_ok=True)
N_SPLITS = 5
# SEEDS = [5, 12]
SEEDS = [0]
N_TRIALS = 50
threshold = 1e-05  # 1e-15

# DATADIR = '../input/lish-moa/'
DATADIR = (
    r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\input\lish-moa"
)
dtype = {"cp_type": "category", "cp_dose": "category"}
index_col = "sig_id"
train = pd.read_csv(f"{DATADIR}/train_features.csv", dtype=dtype, index_col=index_col)
test = pd.read_csv(f"{DATADIR}/test_features.csv", dtype=dtype, index_col=index_col)
train_targets = pd.read_csv(
    f"{DATADIR}/train_targets_scored.csv", dtype=dtype, index_col=index_col
)
# train_targets_nonscored = pd.read_csv(f"{DATADIR}/train_targets_nonscored.csv", dtype=dtype, index_col=index_col)
groups = pd.read_csv(
    f"{DATADIR}/train_drug.csv", dtype=dtype, index_col=index_col, squeeze=True
)
submission = pd.read_csv(
    f"{DATADIR}/sample_submission.csv", dtype=dtype, index_col=index_col
)

if DEBUG:
    np.random.seed(0)  # 乱数シード固定
    # 3クラスのみにする
    _classes = [
        # "sig_id",
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

    print(f"train.shape: {train.shape}")
    return train, train_targets, test


def mean_log_loss(train_targets, oof):
    """マルチラベル全体でlog lossを平均する"""
    scores = []
    for target_col in train_targets.columns:
        oof_clip = np.clip(oof[target_col], threshold, 1 - threshold)
        scores.append(log_loss(train_targets[target_col], oof_clip))
    return np.mean(scores)


def submit(test_pred, test, sample_submission, train_targets, s_csv="submission.csv"):
    for target_col in test_pred.columns:
        test_pred[target_col] = np.clip(
            test_pred[target_col], threshold, 1 - threshold
        )  # mean_log_loss()と同じクリップ
    sample_submission.loc[:, train_targets.columns] = test_pred.values
    sample_submission.loc[test["cp_type"] == 1, train_targets.columns] = 0
    # sample_submission.to_csv(s_csv, index=False)
    sample_submission.to_csv(s_csv)
    return sample_submission


def drug_GroupStratifiedKFold(drug, class1_scored, folds=5, seed=None):
    """薬物およびラベル層別化コード
    https://www.kaggle.com/c/lish-moa/discussion/195195
    - 薬物のみを層別化したい場合は MultilabelStratifiedKFold を KFold に変更したらいいみたい
    Usage:
        class1_scored = pd.DataFrame(Y["5-alpha_reductase_inhibitor"]).reset_index()
        scored = drug_GroupStratifiedKFold(drug, class1_scored, folds=5, seed=42)
        for fold in tqdm(range(5)):
            val_ind = scored[scored["fold"] == fold].index
            trn_ind = scored[scored["fold"] != fold].index
            print(val_ind)
    """
    scored = pd.DataFrame(class1_scored).reset_index()  # sig_idがindexなので列に戻す
    targets = scored.drop(["sig_id"], axis=1).columns  # sig_id列以外がクラス列
    scored = scored.merge(drug, on="sig_id", how="left")

    # LOCATE DRUGS 数が少ない薬(18行以下)は分ける
    vc = scored.drug_id.value_counts()
    vc1 = vc.loc[(vc == 6) | (vc == 12) | (vc == 18)].index.sort_values()
    vc2 = vc.loc[(vc != 6) & (vc != 12) & (vc != 18)].index.sort_values()

    # STRATIFY DRUGS 18X OR LESS
    dct1 = {}
    dct2 = {}
    if seed is None:
        skf = StratifiedKFold(n_splits=folds, shuffle=False)
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    tmp = scored.groupby("drug_id")[targets].mean().loc[vc1]
    for fold, (idxT, idxV) in enumerate(skf.split(tmp, tmp[targets])):
        dd = {k: fold for k in tmp.index[idxV].values}
        dct1.update(dd)

    # STRATIFY DRUGS MORE THAN 18X
    if seed is None:
        skf = StratifiedKFold(n_splits=folds, shuffle=False)
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    tmp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop=True)
    for fold, (idxT, idxV) in enumerate(skf.split(tmp, tmp[targets])):
        dd = {k: fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)

    # ASSIGN FOLDS
    scored["fold"] = scored.drug_id.map(dct1)
    scored.loc[scored.fold.isna(), "fold"] = scored.loc[
        scored.fold.isna(), "sig_id"
    ].map(dct2)
    scored.fold = scored.fold.astype("int8")
    return scored


# --------------------------------------- Feature-Engineering ---------------------------------------


def get_features_gc(train, top_feat_cols=None):
    """g-,c-列の列名取得"""
    if top_feat_cols is None:
        top_feat_cols = train.columns
    features_g = [col for col in top_feat_cols if re.match("g-[0-771]", col)]
    features_c = [col for col in top_feat_cols if re.match("c-[0-99]", col)]
    return features_g, features_c


def get_features(train):
    """特徴量の列名取得"""
    features = train.columns.to_list()
    if "sig_id" in features:
        features.remove("sig_id")
    if "cp_type" in features:
        features.remove("cp_type")
    return features


def scaling(train, test, scaler=RobustScaler(), is_fit_train_only=True):
    """規格化。pcaの後に実行してる。pcaの後だから外れ値にロバストな規格化使ってるみたい"""
    features = get_features(train)
    # scaler = RobustScaler()  # 外れ値に頑健な標準化
    if is_fit_train_only:
        scaler.fit(train[features])
    else:
        scaler.fit(pd.concat([train[features], test[features]], axis=0))
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, get_features(train)


def fe_pca(
    train,
    test,
    n_components_g=70,
    n_components_c=10,
    random_state=123,
    is_fit_train_only=True,
):
    """pcaで圧縮した特徴量追加"""
    from sklearn.decomposition import PCA

    features_g, features_c = get_features_gc(train)

    def create_pca(
        train,
        test,
        features,
        kind="g",
        n_components=n_components_g,
        is_fit_train_only=True,
    ):
        train_ = train[features].copy()
        test_ = test[features].copy()
        pca = PCA(n_components=n_components, random_state=random_state)
        columns = [f"pca_{kind}{i + 1}" for i in range(n_components)]
        if is_fit_train_only:
            # trainだけでpca fitする場合
            train_ = pca.fit_transform(train_)
            train_ = pd.DataFrame(
                pca.fit_transform(train_), index=train.index, columns=columns
            )
            test_ = pd.DataFrame(
                pca.fit_transform(test_), index=test.index, columns=columns
            )
        else:
            data_ = pd.concat([train_, test_], axis=0)
            data = data_.copy()
            data_ = pca.fit_transform(data_)
            data_ = pd.DataFrame(
                pca.fit_transform(data_), index=data.index, columns=columns
            )
            train_ = data_.iloc[: train.shape[0]]
            test_ = data_.iloc[train.shape[0] :]  # .reset_index(drop=True)
        train = pd.concat([train, train_], axis=1)
        test = pd.concat([test, test_], axis=1)
        return train, test

    train, test = create_pca(
        train,
        test,
        features_g,
        kind="g",
        n_components=n_components_g,
        is_fit_train_only=is_fit_train_only,
    )
    train, test = create_pca(
        train,
        test,
        features_c,
        kind="c",
        n_components=n_components_c,
        is_fit_train_only=is_fit_train_only,
    )

    return train, test


def fe_clipping(
    train, test, min_clip=0.01, max_clip=0.99,
):
    """外れ値の特徴量クリップ"""
    features_g, features_c = get_features_gc(train)

    def _clipping(
        train, test, features, min_clip, max_clip,
    ):
        train_ = train[features].copy()
        test_ = test[features].copy()
        df = pd.concat([train_, test_], axis=0)

        p_min = np.quantile(df.loc[:, features], min_clip)
        p_max = np.quantile(df.loc[:, features], max_clip)
        print(f"{min_clip * 100} / {max_clip * 100} % clip : {p_min} / {p_max}")
        # 1％点以下の値は1％点に、99％点以上の値は99％点にclippingする
        df[features] = df[features].clip(p_min, p_max, axis=1)

        train[features] = df.iloc[: train.shape[0]]
        test[features] = df.iloc[train.shape[0] :].reset_index(drop=True)
        return train, test

    train, test = _clipping(train, test, features_g, min_clip, max_clip)
    train, test = _clipping(train, test, features_c, min_clip, max_clip)

    return train, test


def fe_stats(train, test, params=["g", "c"], flag_add=False):
    """統計量の特徴量追加"""

    features_g, features_c = get_features_gc(train)

    for df in [train, test]:
        if "g" in params:
            # df["g_sum"] = df[features_g].sum(axis=1)
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
            # df["c_sum"] = df[features_c].sum(axis=1)
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
            # df["gc_sum"] = df[features_g + features_c].sum(axis=1)
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
    """cの特徴量を2乗した特徴量作成"""
    features_g, features_c = get_features_gc(train)
    for df in [train, test]:
        for feature in features_c:
            df[f"squared_{feature}"] = df[feature] ** 2
    return train, test


def c_abs(train, test):
    """cの特徴量を絶対値とった特徴量作成"""
    features_g, features_c = get_features_gc(train)
    for df in [train, test]:
        for feature in features_c:
            df[f"abs_{feature}"] = np.abs(df[feature])
    return train, test


def g_valid(train, test):
    """gの特徴量の有効フラグ追加
    gの絶対値>2,<2は薬が効いて、0近くだと効いてないらしい
    https://www.kaggle.com/mrbhbs/discussion
    """
    features_g, features_c = get_features_gc(train)
    for df in [train, test]:
        for feature in features_g:
            df[f"{feature}_valid"] = df[feature].apply(
                lambda x: 1.0 if (np.abs(x) > 2) & (np.abs(x) < 2) else 0.0
            )
    return train, test


def fe_noise(train, test, sigma_down_ratio=10.0, seed=123):
    """g-,c-それぞれにノイズ加算"""
    # g-,c-単位で実行
    features_g, features_c = get_features_gc(train)

    def df_add_normal_noise(df, mu, sigma, seed):
        """データフレームにガウシアンノイズを加算する
        ノイズの平均と標準偏差指定必要"""
        np.random.seed(seed)
        noise = np.random.normal(mu, sigma, [df.shape[0], df.shape[1]])
        df = df + noise
        return df

    def _noise(train, test, features, sigma_down_ratio, seed):
        train_ = train[features].copy()
        test_ = test[features].copy()
        df = pd.concat([train_, test_], axis=0)

        mu = np.median(df.values)
        sigma = np.std(df.values) / sigma_down_ratio  # 単純に標準偏差渡すと値変わりすぎるので割り算で減らす
        df = df_add_normal_noise(df, mu, sigma, seed)

        train[features] = df.iloc[: train.shape[0]]
        test[features] = df.iloc[train.shape[0] :].reset_index(drop=True)
        return train, test

    train, test = _noise(train, test, features_g, sigma_down_ratio, seed)
    train, test = _noise(train, test, features_c, sigma_down_ratio, seed)

    return train, test


# ---------------------------------------------------------------------------------------------------


def save_model(model, model_path="model/fold00.model"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path, compress=True)


def load_model(model_path="model/fold00.model"):
    return joblib.load(model_path)


def run_lgb(target_col: str, params, seed):
    """1クラスだけ学習"""
    X_train = train[features]
    y_train = train_targets[target_col]
    X_test = test[features]

    counts = np.empty((N_SPLITS, 1))
    y_preds = []
    f_importances = []
    oof_train = np.zeros((len(X_train),))

    scored = drug_GroupStratifiedKFold(groups, y_train, folds=N_SPLITS, seed=seed)
    for fold_id in tqdm(range(N_SPLITS)):
        valid_index = scored[scored["fold"] == fold_id].index
        train_index = scored[scored["fold"] != fold_id].index
        print(f"\n------------ fold:{fold_id} ------------")
        X_tr = X_train.iloc[train_index, :]
        X_val = X_train.iloc[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]
        counts[fold_id] = y_tr.sum()

        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

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
        counts,
    )


def run_all_target(params, seeds=SEEDS):
    """乱数変えて全クラス学習"""
    _cols = train_targets.columns.to_list()[:]

    oof = train_targets.copy()
    sub = submission.copy()
    oof.loc[:, _cols] = 0.0
    sub.loc[:, _cols] = 0.0
    importance_dfs = []
    counts = []

    for seed in seeds:
        print(f"\n================ seed:{seed} ================")
        for i, target_col in tqdm(enumerate(train_targets.columns)):
            print(f"\n########## {i} target_col:{target_col} ##########")
            _oof, _preds, _importance_df, _counts = run_lgb(target_col, params, seed)
            oof[target_col] += _oof
            sub[target_col] += _preds
            importance_dfs.append(_importance_df)
            counts.append(_counts)
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

    with open(f"{OUTDIR}/counts.pkl", "wb") as f:
        pickle.dump(counts, f)

    with open(f"{OUTDIR}/Y_pred.pkl", "wb") as f:
        pickle.dump(oof, f)

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
    params["max_depth"] = trial.suggest_int("max_depth", 1, 3)  # 7)
    params["num_leaves"] = trial.suggest_int(
        "num_leaves", 2, 4
    )  # 2 ** params["max_depth"])
    params["min_data_in_leaf"] = trial.suggest_int(
        "min_data_in_leaf",
        1,
        max(
            1, int(train.shape[0] * ((N_SPLITS - 1) / N_SPLITS) / params["num_leaves"])
        ),
    )

    params["feature_fraction"] = trial.suggest_discrete_uniform(
        "feature_fraction", 0.1, 1.0, 0.05
    )
    params["lambda_l1"] = trial.suggest_loguniform("lambda_l1", 1e-09, 10.0)
    params["lambda_l2"] = trial.suggest_loguniform("lambda_l2", 1e-09, 10.0)

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

    oof_metric = f"oof_log_loss: {mean_log_loss(train_targets, oof)}"
    with open(f"{OUTDIR}/oof_metric.txt", mode="w") as f:
        f.write(oof_metric)
    print(oof_metric)

    submit(sub, test, submission, train_targets, s_csv=f"{OUTDIR}/submission.csv")


if __name__ == "__main__":
    print(
        f"### start:", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "###"
    )

    train, train_targets, test = mapping_and_filter(train, train_targets, test)
    train, test = fe_clipping(train, test)
    train, test = g_valid(train, test)
    train, test = fe_stats(train, test)
    train, test = fe_noise(train, test)

    # train, test = c_squared(train, test)
    # train, test = fe_pca(
    #    train, test, n_components_g=70, n_components_c=10, random_state=123
    # )
    # train, test, features = scaling(train, test)

    features = get_features(train)
    if "cp_type" in features:
        features.remove("cp_type")
    if "cp_dose" in features:
        features.remove("cp_dose")
    if "cp_time" in features:
        features.remove("cp_time")

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
