import os
import sys
import re
import pickle
import datetime
import logging
import pathlib
import random as rn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler

import tensorflow as tf


# ===================================================================================================
# ----------------------------------------------- Util ----------------------------------------------
# ===================================================================================================


def plot_history(history, figsize=(16, 9)):
    f, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(linestyle=":")
    ax.plot(history.history["loss"], label=f"train")
    ax.plot(history.history["val_loss"], label=f"valid")
    ax.legend()
    plt.savefig("result.png")
    plt.clf()
    plt.close()
    return ax


class Logger:
    def __init__(self, output_dir=None):
        self.general_logger = logging.getLogger("general")
        self.result_logger = logging.getLogger("result")
        stream_handler = logging.StreamHandler()

        # ディレクトリ指定無ければカレントディレクトリにログファイル出す
        output_dir = pathlib.Path.cwd() if output_dir is None else output_dir
        file_general_handler = logging.FileHandler(
            os.path.join(output_dir, "general.log")
        )
        file_result_handler = logging.FileHandler(
            os.path.join(output_dir, "result.log")
        )

        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        """時刻をつけてコンソールとgeneral.log（ログファイル）に文字列書き込み"""
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    def result_scores(self, run_name, scores):
        """
        計算結果をコンソールとresult.log（cv結果用ログファイル）に書き込み
        parms: run_name: 実行したcvの名前
        parms: scores: cv scoreのリスト。result.logには平均値も書く
        """
        dic = dict()
        dic["name"] = run_name
        dic["score"] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f"score{i}"] = score
        self.result(self.to_ltsv(dic))

    def result(self, message):
        """コンソールとresult.logに文字列書き込み"""
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        """コンソールとresult.logに辞書データ書き込み"""
        self.result(self.to_ltsv(dic))

    def now_string(self):
        """時刻返すだけ"""
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_ltsv(self, dic):
        """辞書を文字列に変えるだけ"""
        return "\t".join(["{}:{}".format(key, value) for key, value in dic.items()])


# ===================================================================================================
# --------------------------------------- Feature-Engineering ---------------------------------------
# ===================================================================================================


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


# =========================================== add ===============================================


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

    return train, test, get_features(train)


def fe_quantile_transformer(train, test, n_quantiles=100, random_state=0):
    """QuantileTransformerで特徴量の分布を一様にする(RankGauss)"""
    from sklearn.preprocessing import QuantileTransformer

    features_g, features_c = get_features_gc(train)
    for col in features_g + features_c:

        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            random_state=random_state,
            output_distribution="normal",
        )

        vec_len = len(train[col].values)
        raw_vec = train[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)
        train[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]

        vec_len_test = len(test[col].values)
        test[col] = transformer.transform(
            test[col].values.reshape(vec_len_test, 1)
        ).reshape(1, vec_len_test)[0]

    return train, test


def fe_variance_threshold(train, test, features, threshold=0.8, is_fit_train_only=True):
    """分散低い特徴量削除"""
    from sklearn.feature_selection import VarianceThreshold

    var_thresh = VarianceThreshold(threshold)
    if is_fit_train_only:
        data = train.copy()
    else:
        data = train.append(test)
    # g-,c-列について適用
    data_transformed = var_thresh.fit_transform(data.loc[:, features])
    data_transformed = pd.DataFrame(data_transformed, index=data.index)

    train_transformed = data_transformed[: train.shape[0]]
    test_transformed = data_transformed[-test.shape[0] :]

    # train = pd.DataFrame(
    #    train[["sig_id", "cp_type", "cp_time", "cp_dose"]].values.reshape(-1, 4),
    #    columns=["sig_id", "cp_type", "cp_time", "cp_dose"],
    # )
    # train = pd.concat([train, pd.DataFrame(train_transformed)], axis=1)
    # test = pd.DataFrame(
    #    test[["sig_id", "cp_type", "cp_time", "cp_dose"]].values.reshape(-1, 4),
    #    columns=["sig_id", "cp_type", "cp_time", "cp_dose"],
    # )
    # test = pd.concat([test, pd.DataFrame(test_transformed)], axis=1)
    # return train, test, get_features(train)
    return train_transformed, test_transformed


def fe_cluster(
    train,
    test,
    n_clusters_g=35,
    n_clusters_c=5,
    random_state=123,
    is_fit_train_only=True,
):
    """KMeansで特徴量作成"""
    from sklearn.cluster import KMeans

    features_g, features_c = get_features_gc(train)

    def create_cluster(train, test, features, kind="g", n_clusters=n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        if is_fit_train_only:
            data = train.copy()
        else:
            data = pd.concat([train_, test_], axis=0)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)
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
    features_g, features_c = get_features_gc(train)
    for df in [train, test]:
        for feature in features_g:
            df[f"valid_{feature}"] = df[feature].apply(
                lambda x: 1.0 if (np.abs(x) > 2) & (np.abs(x) < 2) else 0.0
            )
    return train, test, get_features(train)


def c_squared(train, test, features_c=None):
    """cの特徴量を2乗した特徴量作成"""
    features_g, features_c = get_features_gc(train)
    for df in [train, test]:
        for feature in features_c:
            df[f"squared_{feature}"] = df[feature] ** 2
    return train, test, get_features(train)


def c_abs(train, test, features_c=None):
    """cの特徴量を絶対値とった特徴量作成"""
    features_g, features_c = get_features_gc(train)
    for df in [train, test]:
        for feature in features_c:
            df[f"abs_{feature}"] = np.abs(df[feature])
    return train, test, get_features(train)


# ===================================================================================================
# -------------------------------------------- tensorflow -------------------------------------------
# ===================================================================================================

# https://arxiv.org/abs/2002.08709
def flooding(loss, b=0.01):
    def flood(y_true, y_pred):
        return tf.abs(loss(y_true, y_pred) - b) + b

    return flood


# https://arxiv.org/abs/1905.04899
class Cutmix(tf.keras.utils.Sequence):
    def __init__(self, X, y=None, batch_size=32, alpha=1.0):
        self.X = np.asarray(X)
        if y is None:
            self.y = y
        else:
            self.y = np.asarray(y)
        self.batch_size = batch_size
        self.alpha = alpha

    def __getitem__(self, i):
        X_batch = self.X[i * self.batch_size : (i + 1) * self.batch_size]
        n_samples, n_features = self.X.shape
        batch_size = X_batch.shape[0]
        shuffle = np.random.choice(n_samples, batch_size)
        l = np.random.beta(self.alpha, self.alpha)
        mask = np.random.choice([0.0, 1.0], size=n_features, p=[1.0 - l, l])
        X_shuffle = self.X[shuffle]
        X_batch = mask * X_batch + (1.0 - mask) * X_shuffle
        if self.y is None:
            return X_batch, None
        y_batch = self.y[i * self.batch_size : (i + 1) * self.batch_size]
        y_shuffle = self.y[shuffle]
        y_batch = l * y_batch + (1.0 - l) * y_shuffle
        return X_batch, y_batch

    def __len__(self):
        n_samples = self.X.shape[0]
        return int(np.ceil(n_samples / self.batch_size))


# https://arxiv.org/abs/1811.05850
class DropActivation(tf.keras.layers.Layer):
    def __init__(self, activation="relu", rate=0.05, **kwargs):
        super().__init__(**kwargs)
        if callable(activation):
            self.activation = activation
        else:
            self.activation = tf.keras.activations.get(activation)
        self.rate = rate
        return

    def call(self, x, training=None):
        def outputs_in_train():
            mask = tf.keras.backend.ones_like(x)
            mask = tf.keras.backend.dropout(mask, level=self.rate)
            return (1.0 - mask) * x + mask * self.activation(x)

        def outputs_in_test():
            return self.rate * x + (1.0 - self.rate) * self.activation(x)

        return tf.keras.backend.in_train_phase(
            outputs_in_train, outputs_in_test, training=training
        )

    def get_config(self):
        config = super().get_config()
        return dict(activation=self.activation, rate=self.rate, **config)


# https://arxiv.org/abs/1710.09412
class Mixup(tf.keras.utils.Sequence):
    def __init__(self, X, y=None, batch_size=32, alpha=0.2):
        self.X = np.asarray(X)
        if y is None:
            self.y = y
        else:
            self.y = np.asarray(y)
        self.batch_size = batch_size
        self.alpha = alpha

    def __getitem__(self, i):
        X_batch = self.X[i * self.batch_size : (i + 1) * self.batch_size]
        n_samples = self.X.shape[0]
        batch_size = X_batch.shape[0]
        shuffle = np.random.choice(n_samples, batch_size)
        l = np.random.beta(self.alpha, self.alpha)
        X_shuffle = self.X[shuffle]
        X_batch = l * X_batch + (1.0 - l) * X_shuffle
        if self.y is None:
            return X_batch, None
        y_batch = self.y[i * self.batch_size : (i + 1) * self.batch_size]
        y_shuffle = self.y[shuffle]
        y_batch = l * y_batch + (1.0 - l) * y_shuffle
        return X_batch, y_batch

    def __len__(self):
        n_samples = self.X.shape[0]
        return int(np.ceil(n_samples / self.batch_size))


# ===================================================================================================
# ------------------------------------------ Main notebook ------------------------------------------
# ===================================================================================================


def build_callbacks(
    model_path, factor=0.1, mode="auto", monitor="val_loss", patience=0, verbose=0
):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        mode=mode, monitor=monitor, patience=patience, verbose=verbose
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, mode=mode, monitor=monitor, save_best_only=True, verbose=verbose
    )
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        factor=factor, monitor=monitor, mode=mode, verbose=verbose
    )

    return [early_stopping, model_checkpoint, reduce_lr_on_plateau]


def predict(model, X, n_iter=1, **predict_params):
    if isinstance(X, tf.keras.utils.Sequence):
        func = model.predict_generator
    else:
        func = model.predict

    for i in range(n_iter):
        if i == 0:
            y_pred = func(X, **predict_params) / n_iter
        else:
            y_pred += func(X, **predict_params) / n_iter

    return y_pred


def set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)

    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    graph = tf.compat.v1.get_default_graph()
    session_conf = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=1, intra_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(graph=graph, config=session_conf)

    tf.compat.v1.keras.backend.set_session(sess)


def score(Y, Y_pred):
    _, n_classes = Y.shape

    losses = []

    for j in range(n_classes):
        loss = log_loss(Y.iloc[:, j], Y_pred.iloc[:, j], labels=[0, 1])

        losses.append(loss)

    return np.mean(losses)


def load_data():
    if "jupyter_notebook" in os.getcwd():
        # load
        dtype = {"cp_type": "category", "cp_dose": "category"}
        index_col = "sig_id"

        sys.path.append(
            r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\code"
        )
        import datasets

        DATADIR = datasets.DATADIR

        groups = pd.read_csv(
            f"{DATADIR}/train_drug.csv", dtype=dtype, index_col=index_col, squeeze=True
        )
        train_features = pd.read_csv(
            f"{DATADIR}/train_features.csv", dtype=dtype, index_col=index_col
        )
        X_test = pd.read_csv(
            f"{DATADIR}/test_features.csv", dtype=dtype, index_col=index_col
        )
        X = train_features.select_dtypes("number")
        Y_nonscored = pd.read_csv(
            f"{DATADIR}/train_targets_nonscored.csv", index_col=index_col
        )
        Y = pd.read_csv(f"{DATADIR}/train_targets_scored.csv", index_col=index_col)

        columns = Y.columns

    else:
        # load
        dtype = {"cp_type": "category", "cp_dose": "category"}
        index_col = "sig_id"

        groups = pd.read_csv(
            f"../input/lish-moa/train_drug.csv",
            dtype=dtype,
            index_col=index_col,
            squeeze=True,
        )
        train_features = pd.read_csv(
            "../input/lish-moa/train_features.csv", dtype=dtype, index_col=index_col
        )
        X_test = pd.read_csv(
            "../input/lish-moa/test_features.csv", dtype=dtype, index_col=index_col
        )
        X = train_features.select_dtypes("number")
        Y_nonscored = pd.read_csv(
            "../input/lish-moa/train_targets_nonscored.csv", index_col=index_col
        )
        Y = pd.read_csv(
            "../input/lish-moa/train_targets_scored.csv", index_col=index_col
        )

        columns = Y.columns

    return X, Y, Y_nonscored, train_features, columns, groups, X_test
