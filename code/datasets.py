"""Load Data + Feature-Engineering"""
import sys
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from params import DATADIR, FOLDS, ITERATIVE_STRATIFICATION

sys.path.append(ITERATIVE_STRATIFICATION)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# ===================================================================================================
# ----------------------------------------------- CV ------------------------------------------------
# ===================================================================================================


def drug_MultilabelStratifiedKFold(folds=FOLDS, seed=None, scored=None):
    """薬物およびマルチラベル層別化コード
    https://www.kaggle.com/c/lish-moa/discussion/195195
    - 薬物のみを層別化したい場合は MultilabelStratifiedKFold を KFold に変更したらいいみたい
    Usage:
        scored = drug_MultilabelStratifiedKFold()
        for fold in tqdm(range(FOLDS)):
            val_ind = scored[scored["fold"] == fold].index
            trn_ind = scored[scored["fold"] != fold].index
    """
    # LOAD FILES
    drug = pd.read_csv(f"{DATADIR}/train_drug.csv")
    if scored is None:
        scored = pd.read_csv(f"{DATADIR}/train_targets_scored.csv")
    targets = scored.drop(["sig_id"], axis=1).columns  # sig_id列以外がクラス列
    scored = scored.merge(drug, on="sig_id", how="left")

    # LOCATE DRUGS 数が少ない薬(18行以下)は分ける
    vc = scored.drug_id.value_counts()
    vc1 = vc.loc[vc <= 18].index.sort_values()
    vc2 = vc.loc[vc > 18].index.sort_values()

    # STRATIFY DRUGS 18X OR LESS 数が少ない薬(18行以下)をcvに分ける
    dct1 = {}
    dct2 = {}
    if seed is None:
        skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=False)
    else:
        skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    tmp = scored.groupby("drug_id")[targets].mean().loc[vc1]
    for fold, (idxT, idxV) in enumerate(skf.split(tmp, tmp[targets])):
        dd = {k: fold for k in tmp.index[idxV].values}
        dct1.update(dd)

    # STRATIFY DRUGS MORE THAN 18X 数が多い薬(18行以上)をcvに分ける
    if seed is None:
        skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=False)
    else:
        skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
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


# ===================================================================================================
# ---------------------------------------------- Data -----------------------------------------------
# ===================================================================================================


def load_orig_data():
    train = pd.read_csv(f"{DATADIR}/train_features.csv")
    train_targets = pd.read_csv(f"{DATADIR}/train_targets_scored.csv")
    train_targets_nonscored = pd.read_csv(f"{DATADIR}/train_targets_nonscored.csv")
    test = pd.read_csv(f"{DATADIR}/test_features.csv")
    sample_submission = pd.read_csv(f"{DATADIR}/sample_submission.csv")
    train_drug = pd.read_csv(f"{DATADIR}/train_drug.csv")
    return (
        train,
        train_targets,
        test,
        sample_submission,
        train_targets_nonscored,
        train_drug,
    )


def mapping_and_filter(
    train,
    train_targets,
    test,
    train_targets_nonscored,
    is_del_ctl=False,
    is_conat_nonscore=False,
    is_del_noise_drug=False,
):
    """前処理"""
    cp_type = {"trt_cp": 0, "ctl_vehicle": 1}
    cp_dose = {"D1": 0, "D2": 1}
    for df in [train, test]:
        df["cp_type"] = df["cp_type"].map(cp_type)
        df["cp_dose"] = df["cp_dose"].map(cp_dose)

    # 完全なoofにするためデータ抜くのやめる 20201028
    if is_del_ctl:
        # ctl_vehicleは必ず0なので学習データから除く
        train_targets = train_targets[train["cp_type"] == 0].reset_index(drop=True)
        train_targets_nonscored = train_targets_nonscored[
            train["cp_type"] == 0
        ].reset_index(drop=True)
        train = train[train["cp_type"] == 0].reset_index(drop=True)

    # 同じ用量と時間ですが、遺伝子と細胞のデータはすべて著しく異なる行削除 20201105
    # https://www.kaggle.com/c/lish-moa/discussion/195245
    if is_del_noise_drug:
        train_drug = pd.read_csv(f"{DATADIR}/train_drug.csv")
        for d_id in ["87d714366"]:
            train = train[train_drug["drug_id"] != d_id].reset_index(drop=True)
            train_targets = train_targets[train_drug["drug_id"] != d_id].reset_index(
                drop=True
            )
            train_targets_nonscored = train_targets_nonscored[
                train_drug["drug_id"] != d_id
            ].reset_index(drop=True)

    # train_targets.drop(["sig_id"], inplace=True, axis=1)  # drug_MultilabelStratifiedKFold で使うからsig_id列残す
    train_targets_nonscored.drop(["sig_id"], inplace=True, axis=1)  # sig_id列はidなので不要

    # nonscored と連結
    if is_conat_nonscore:
        train_targets_nonscored = train_targets_nonscored.loc[
            :, ~(train_targets_nonscored.nunique() == 1)
        ]  # nonscoredすべて0の列削除
        train_targets = pd.concat([train_targets_nonscored, train_targets], axis=1)

    return train, train_targets, test, train_targets_nonscored


def get_features_gc(train, top_feat_cols=None):
    """g-,c-列の列名取得"""
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


# ===================================================================================================
# --------------------------------------- Feature-Engineering ---------------------------------------
# ===================================================================================================


def scaling(train, test, scaler=RobustScaler()):
    """規格化。pcaの後に実行してる。pcaの後だから外れ値にロバストな規格化使ってるみたい"""
    features = train.columns[2:]
    # scaler = RobustScaler()  # 外れ値に頑健な標準化
    scaler.fit(pd.concat([train[features], test[features]], axis=0))
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, get_features(train)


def fe_pca(
    train,
    test,
    features_g=None,
    features_c=None,
    n_components_g=70,
    n_components_c=10,
    SEED=123,
    is_fit_train_only=False,
):
    """pcaで圧縮した特徴量追加"""

    # 特徴量分けているが大区分がgとcの2区分あるので、それぞれでpca
    features_g = list(train.columns[4:776]) if features_g is None else features_g
    features_c = list(train.columns[776:876]) if features_c is None else features_c

    def create_pca(
        train,
        test,
        features,
        kind="g",
        n_components=n_components_g,
        is_fit_train_only=False,
    ):
        train_ = train[features].copy()
        test_ = test[features].copy()
        pca = PCA(n_components=n_components, random_state=SEED)
        columns = [f"pca_{kind}{i + 1}" for i in range(n_components)]
        if is_fit_train_only:
            # trainだけでpca fitする場合
            train_ = pca.fit_transform(train_)
            train_ = pd.DataFrame(train_, columns=columns)
            test_ = pca.transform(test_)
            test_ = pd.DataFrame(test_, columns=columns)
        else:
            data = pd.concat([train_, test_], axis=0)
            data = pca.fit_transform(data)
            data = pd.DataFrame(data, columns=columns)
            train_ = data.iloc[: train.shape[0]]
            test_ = data.iloc[train.shape[0] :].reset_index(drop=True)
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

    return train, test, get_features(train)


def fe_clipping(
    train, test, features_g=None, features_c=None, min_clip=0.01, max_clip=0.99,
):
    """外れ値の特徴量クリップ"""
    # g-,c-単位で実行
    features_g = list(train.columns[4:776]) if features_g is None else features_g
    features_c = list(train.columns[776:876]) if features_c is None else features_c

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


def fe_stats(
    train,
    test,
    features_g=None,
    features_c=None,
    params=["g", "c", "gc"],
    flag_add=False,
):
    """統計量の特徴量追加"""
    features_g = list(train.columns[4:776]) if features_g is None else features_g
    features_c = list(train.columns[776:876]) if features_c is None else features_c

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


def fe_quantile_transformer(train, test, features_g=None, features_c=None):
    """QuantileTransformerで特徴量の分布を一様にする(RankGauss)"""
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
    # この変換は最も頻繁な値を分散させる傾向があります。また、（わずかな）外れ値の影響も軽減します。したがって、これは堅牢な前処理スキーム
    from sklearn.preprocessing import QuantileTransformer

    features_g = list(train.columns[4:776]) if features_g is None else features_g
    features_c = list(train.columns[776:876]) if features_c is None else features_c

    for col in features_g + features_c:

        transformer = QuantileTransformer(
            n_quantiles=100, random_state=0, output_distribution="normal"
        )
        vec_len = len(train[col].values)
        vec_len_test = len(test[col].values)
        raw_vec = train[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test[col] = transformer.transform(
            test[col].values.reshape(vec_len_test, 1)
        ).reshape(1, vec_len_test)[0]

    return train, test


def fe_variance_threshold(train, test, threshold=0.8):
    """分散で低い特徴量削除"""
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html?highlight=variancethreshold#sklearn.feature_selection.VarianceThreshold
    # トレーニングセットの分散がこのしきい値よりも低い特徴は削除
    from sklearn.feature_selection import VarianceThreshold

    var_thresh = VarianceThreshold(threshold)
    data = train.append(test)
    # g-,c-列について適用
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

    train_transformed = data_transformed[: train.shape[0]]
    test_transformed = data_transformed[-test.shape[0] :]

    train = pd.DataFrame(
        train[["sig_id", "cp_type", "cp_time", "cp_dose"]].values.reshape(-1, 4),
        columns=["sig_id", "cp_type", "cp_time", "cp_dose"],
    )
    train = pd.concat([train, pd.DataFrame(train_transformed)], axis=1)
    test = pd.DataFrame(
        test[["sig_id", "cp_type", "cp_time", "cp_dose"]].values.reshape(-1, 4),
        columns=["sig_id", "cp_type", "cp_time", "cp_dose"],
    )
    test = pd.concat([test, pd.DataFrame(test_transformed)], axis=1)

    return train, test, get_features(train)


def fe_cluster(
    train,
    test,
    features_g=None,
    features_c=None,
    n_clusters_g=35,
    n_clusters_c=5,
    SEED=123,
):
    """KMeansで特徴量作成"""
    from sklearn.cluster import KMeans

    features_g = list(train.columns[4:776]) if features_g is None else features_g
    features_c = list(train.columns[776:876]) if features_c is None else features_c

    def create_cluster(train, test, features, kind="g", n_clusters=n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis=0)
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(data)
        train[f"clusters_{kind}"] = kmeans.labels_[: train.shape[0]]
        test[f"clusters_{kind}"] = kmeans.labels_[train.shape[0] :]
        train = pd.get_dummies(train, columns=[f"clusters_{kind}"])
        test = pd.get_dummies(test, columns=[f"clusters_{kind}"])
        return train, test

    train, test = create_cluster(
        train, test, features_g, kind="g", n_clusters=n_clusters_g
    )
    train, test = create_cluster(
        train, test, features_c, kind="c", n_clusters=n_clusters_c
    )

    return train, test, get_features(train)


def g_valid(train, test, features_g=None):
    """gの特徴量の有効フラグ追加
    gの絶対値>2,<2は薬が効いて、0近くだと効いてないらしい
    https://www.kaggle.com/mrbhbs/discussion
    """
    features_g = list(train.columns[4:776]) if features_g is None else features_g
    for df in [train, test]:
        for feature in features_g:
            df[f"{feature}_valid"] = df[feature].apply(
                lambda x: 1.0 if (np.abs(x) > 2) & (np.abs(x) < 2) else 0.0
            )
    return train, test, get_features(train)


def g_squared(train, test, features_g=None):
    """gの特徴量を2乗した特徴量作成"""
    features_g = list(train.columns[4:776]) if features_g is None else features_g
    for df in [train, test]:
        for feature in features_g:
            df[f"{feature}_squared"] = df[feature] ** 2
    return train, test, get_features(train)


def g_binary(train, test, features_g=None):
    """gを正負で2値化した特徴量作成"""
    features_g = list(train.columns[4:776]) if features_g is None else features_g
    for df in [train, test]:
        for feature in features_g:
            df[f"{feature}_binary"] = df[feature].map(lambda x: 1.0 if x > 0.0 else 0.0)
    return train, test, get_features(train)


def g_abs(train, test, features_g=None):
    """gの特徴量を絶対値とった特徴量作成"""
    features_g = list(train.columns[4:776]) if features_g is None else features_g
    for df in [train, test]:
        for feature in features_g:
            df[f"{feature}_abs"] = np.abs(df[feature])
    return train, test, get_features(train)


def c_squared(train, test, features_c=None):
    """cの特徴量を2乗した特徴量作成"""
    features_c = list(train.columns[776:876]) if features_c is None else features_c
    for df in [train, test]:
        for feature in features_c:
            df[f"{feature}_squared"] = df[feature] ** 2
    return train, test, get_features(train)


def c_binary(train, test, features_c=None):
    """cを正負で2値化した特徴量作成"""
    features_c = list(train.columns[776:876]) if features_c is None else features_c
    for df in [train, test]:
        for feature in features_c:
            df[f"{feature}_binary"] = df[feature].map(lambda x: 1.0 if x > 0.0 else 0.0)
    return train, test, get_features(train)


def c_abs(train, test, features_c=None):
    """cの特徴量を絶対値とった特徴量作成"""
    features_c = list(train.columns[776:876]) if features_c is None else features_c
    for df in [train, test]:
        for feature in features_c:
            df[f"{feature}_abs"] = np.abs(df[feature])
    return train, test, get_features(train)


def drop_col(train, test, col="cp_type"):
    """特徴量1列削除"""
    train = train.drop(col, axis=1)
    test = test.drop(col, axis=1)
    return train, test, get_features(train)


def fe_ctl_mean(
    train,
    test,
    features_g=None,
    features_c=None,
    gc_flg="gc",
    is_mean=True,
    is_ratio=True,
):
    """cp_type=ctl_vehicle のコントロールのレコードの平均との差、比率の特徴量追加"""
    features_g = list(train.columns[4:776]) if features_g is None else features_g
    features_c = list(train.columns[776:876]) if features_c is None else features_c
    features_gc = features_g + features_c
    if gc_flg == "g":
        features_gc = features_g
    elif gc_flg == "c":
        features_gc = features_c

    df = pd.concat([train, test], axis=0)

    # コントロールのレコードの条件ごとに平均出す
    dict_ctl = {}
    for dose in [0, 1]:
        for time in [24, 48, 72]:
            df_ctl = df[
                (df["cp_type"] == 1) & (df["cp_time"] == time) & (df["cp_dose"] == dose)
            ]
            if is_mean:
                ctl_mean = df_ctl[features_gc].apply(lambda x: np.mean(x), axis=0)
                dict_ctl[f"ctl_mean_{dose}_{time}"] = ctl_mean
            else:
                ctl_median = df_ctl[features_gc].apply(lambda x: np.median(x), axis=0)
                dict_ctl[f"ctl_median_{dose}_{time}"] = ctl_median

    # 平均との差、比
    for k, v in dict_ctl.items():
        for fe, me in tqdm(zip(features_gc, v)):
            if is_ratio:
                df[f"{fe}_ratio_{k}"] = df[fe] / me
            else:
                df[f"{fe}_diff_{k}"] = df[fe] - me

    train_ = df.iloc[: train.shape[0]]
    test_ = df.iloc[train.shape[0] :].reset_index(drop=True)
    features = train_.columns[2:]

    return train_, test_, features


def targets_gather(train_targets, is_enc=False):
    """目的変数列を1列にまとめてラベルエンコディング"""
    _targets = train_targets.copy()
    _targets["targets"] = ""
    for ii in range(len(train_targets.columns)):
        _targets["targets"] += _targets.iloc[:, ii].astype("str") + "_"

    _df = None
    if is_enc:
        _targets["targets"], uni = pd.factorize(_targets["targets"])
        _df = pd.DataFrame({"label": list(range(len(uni))), "value": uni})
        # display(_df)

    return _targets["targets"], _df
