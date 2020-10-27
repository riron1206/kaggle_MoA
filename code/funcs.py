"""
メソッド集
構造変えた複数のMLPアンサンブル + 統計量の特徴量追加 + pcaで圧縮した特徴量追加 + 特徴選択 + 予測値をクリップ + label-smoothingしたコード など
参考:
- https://www.kaggle.com/ragnar123/moa-dnn-feature-engineering
"""

import datetime
import logging
import os
import pathlib
import sys
import traceback
import random
import warnings

# sys.path.append('../input/iterative-stratification/iterative-stratification-master')
sys.path.append(r"C:\Users\81908\Git\iterative-stratification")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from sklearn.metrics import log_loss, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

# Adamを改良したAdaBelief https://github.com/juntang-zhuang/Adabelief-Optimizer
# from adabelief_tf import AdaBeliefOptimizer

warnings.filterwarnings("ignore")

FOLDS = 5  # cvの数
# Number of epochs to train each model
EPOCHS = 30
# Batch size
BATCH_SIZE = 124
# Learning rate
LR = 0.001
# Verbosity
VERBOSE = 0

# Seed for deterministic results
SEEDS1 = [1]
SEEDS2 = [8]
SEEDS3 = [15]
SEEDS4 = [22]
SEEDS5 = [29]
# SEEDS1 = [1, 2, 3, 4, 5, 6, 7]
# SEEDS2 = [8, 9, 10, 11, 12, 13, 14]
# SEEDS3 = [15, 16, 17, 18, 19, 20, 21]
# SEEDS4 = [22, 23, 24, 25, 26, 27, 28]
# SEEDS5 = [29, 30, 31, 32, 33, 34, 35]

# Got this predictors from public kernels for the resnet type model
# 特徴量選択しておくみたい public kerelでやってるのパクリらしい
start_predictors = [
    "g-0",
    "g-7",
    "g-8",
    "g-10",
    "g-13",
    "g-17",
    "g-20",
    "g-22",
    "g-24",
    "g-26",
    "g-28",
    "g-29",
    "g-30",
    "g-31",
    "g-32",
    "g-34",
    "g-35",
    "g-36",
    "g-37",
    "g-38",
    "g-39",
    "g-41",
    "g-46",
    "g-48",
    "g-50",
    "g-51",
    "g-52",
    "g-55",
    "g-58",
    "g-59",
    "g-61",
    "g-62",
    "g-63",
    "g-65",
    "g-66",
    "g-67",
    "g-68",
    "g-70",
    "g-72",
    "g-74",
    "g-75",
    "g-79",
    "g-83",
    "g-84",
    "g-85",
    "g-86",
    "g-90",
    "g-91",
    "g-94",
    "g-95",
    "g-96",
    "g-97",
    "g-98",
    "g-100",
    "g-102",
    "g-105",
    "g-106",
    "g-112",
    "g-113",
    "g-114",
    "g-116",
    "g-121",
    "g-123",
    "g-126",
    "g-128",
    "g-131",
    "g-132",
    "g-134",
    "g-135",
    "g-138",
    "g-139",
    "g-140",
    "g-142",
    "g-144",
    "g-145",
    "g-146",
    "g-147",
    "g-148",
    "g-152",
    "g-155",
    "g-157",
    "g-158",
    "g-160",
    "g-163",
    "g-164",
    "g-165",
    "g-170",
    "g-173",
    "g-174",
    "g-175",
    "g-177",
    "g-178",
    "g-181",
    "g-183",
    "g-185",
    "g-186",
    "g-189",
    "g-192",
    "g-194",
    "g-195",
    "g-196",
    "g-197",
    "g-199",
    "g-201",
    "g-202",
    "g-206",
    "g-208",
    "g-210",
    "g-213",
    "g-214",
    "g-215",
    "g-220",
    "g-226",
    "g-228",
    "g-229",
    "g-235",
    "g-238",
    "g-241",
    "g-242",
    "g-243",
    "g-244",
    "g-245",
    "g-248",
    "g-250",
    "g-251",
    "g-254",
    "g-257",
    "g-259",
    "g-261",
    "g-266",
    "g-270",
    "g-271",
    "g-272",
    "g-275",
    "g-278",
    "g-282",
    "g-287",
    "g-288",
    "g-289",
    "g-291",
    "g-293",
    "g-294",
    "g-297",
    "g-298",
    "g-301",
    "g-303",
    "g-304",
    "g-306",
    "g-308",
    "g-309",
    "g-310",
    "g-311",
    "g-314",
    "g-315",
    "g-316",
    "g-317",
    "g-320",
    "g-321",
    "g-322",
    "g-327",
    "g-328",
    "g-329",
    "g-332",
    "g-334",
    "g-335",
    "g-336",
    "g-337",
    "g-339",
    "g-342",
    "g-344",
    "g-349",
    "g-350",
    "g-351",
    "g-353",
    "g-354",
    "g-355",
    "g-357",
    "g-359",
    "g-360",
    "g-364",
    "g-365",
    "g-366",
    "g-367",
    "g-368",
    "g-369",
    "g-374",
    "g-375",
    "g-377",
    "g-379",
    "g-385",
    "g-386",
    "g-390",
    "g-392",
    "g-393",
    "g-400",
    "g-402",
    "g-406",
    "g-407",
    "g-409",
    "g-410",
    "g-411",
    "g-414",
    "g-417",
    "g-418",
    "g-421",
    "g-423",
    "g-424",
    "g-427",
    "g-429",
    "g-431",
    "g-432",
    "g-433",
    "g-434",
    "g-437",
    "g-439",
    "g-440",
    "g-443",
    "g-449",
    "g-458",
    "g-459",
    "g-460",
    "g-461",
    "g-464",
    "g-467",
    "g-468",
    "g-470",
    "g-473",
    "g-477",
    "g-478",
    "g-479",
    "g-484",
    "g-485",
    "g-486",
    "g-488",
    "g-489",
    "g-491",
    "g-494",
    "g-496",
    "g-498",
    "g-500",
    "g-503",
    "g-504",
    "g-506",
    "g-508",
    "g-509",
    "g-512",
    "g-522",
    "g-529",
    "g-531",
    "g-534",
    "g-539",
    "g-541",
    "g-546",
    "g-551",
    "g-553",
    "g-554",
    "g-559",
    "g-561",
    "g-562",
    "g-565",
    "g-568",
    "g-569",
    "g-574",
    "g-577",
    "g-578",
    "g-586",
    "g-588",
    "g-590",
    "g-594",
    "g-595",
    "g-596",
    "g-597",
    "g-599",
    "g-600",
    "g-603",
    "g-607",
    "g-615",
    "g-618",
    "g-619",
    "g-620",
    "g-625",
    "g-628",
    "g-629",
    "g-632",
    "g-634",
    "g-635",
    "g-636",
    "g-638",
    "g-639",
    "g-641",
    "g-643",
    "g-644",
    "g-645",
    "g-646",
    "g-647",
    "g-648",
    "g-663",
    "g-664",
    "g-665",
    "g-668",
    "g-669",
    "g-670",
    "g-671",
    "g-672",
    "g-673",
    "g-674",
    "g-677",
    "g-678",
    "g-680",
    "g-683",
    "g-689",
    "g-691",
    "g-693",
    "g-695",
    "g-701",
    "g-702",
    "g-703",
    "g-704",
    "g-705",
    "g-706",
    "g-708",
    "g-711",
    "g-712",
    "g-720",
    "g-721",
    "g-723",
    "g-724",
    "g-726",
    "g-728",
    "g-731",
    "g-733",
    "g-738",
    "g-739",
    "g-742",
    "g-743",
    "g-744",
    "g-745",
    "g-749",
    "g-750",
    "g-752",
    "g-760",
    "g-761",
    "g-764",
    "g-766",
    "g-768",
    "g-770",
    "g-771",
    "c-0",
    "c-1",
    "c-2",
    "c-3",
    "c-4",
    "c-5",
    "c-6",
    "c-7",
    "c-8",
    "c-9",
    "c-10",
    "c-11",
    "c-12",
    "c-13",
    "c-14",
    "c-15",
    "c-16",
    "c-17",
    "c-18",
    "c-19",
    "c-20",
    "c-21",
    "c-22",
    "c-23",
    "c-24",
    "c-25",
    "c-26",
    "c-27",
    "c-28",
    "c-29",
    "c-30",
    "c-31",
    "c-32",
    "c-33",
    "c-34",
    "c-35",
    "c-36",
    "c-37",
    "c-38",
    "c-39",
    "c-40",
    "c-41",
    "c-42",
    "c-43",
    "c-44",
    "c-45",
    "c-46",
    "c-47",
    "c-48",
    "c-49",
    "c-50",
    "c-51",
    "c-52",
    "c-53",
    "c-54",
    "c-55",
    "c-56",
    "c-57",
    "c-58",
    "c-59",
    "c-60",
    "c-61",
    "c-62",
    "c-63",
    "c-64",
    "c-65",
    "c-66",
    "c-67",
    "c-68",
    "c-69",
    "c-70",
    "c-71",
    "c-72",
    "c-73",
    "c-74",
    "c-75",
    "c-76",
    "c-77",
    "c-78",
    "c-79",
    "c-80",
    "c-81",
    "c-82",
    "c-83",
    "c-84",
    "c-85",
    "c-86",
    "c-87",
    "c-88",
    "c-89",
    "c-90",
    "c-91",
    "c-92",
    "c-93",
    "c-94",
    "c-95",
    "c-96",
    "c-97",
    "c-98",
    "c-99",
]


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


def load_orig_data():
    # train = pd.read_csv("../input/lish-moa/train_features.csv")
    # train_targets = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
    # test = pd.read_csv("../input/lish-moa/test_features.csv")
    # sample_submission = pd.read_csv("../input/lish-moa/sample_submission.csv")
    DATADIR = (
        r"C:\Users\81908\jupyter_notebook\poetry_work\tf23\01_MoA_compe\input\lish-moa"
    )
    train = pd.read_csv(f"{DATADIR}/train_features.csv")
    train_targets = pd.read_csv(f"{DATADIR}/train_targets_scored.csv")
    test = pd.read_csv(f"{DATADIR}/test_features.csv")
    sample_submission = pd.read_csv(f"{DATADIR}/sample_submission.csv")

    return train, train_targets, test, sample_submission


# Function to seed everything
def seed_everything(seed):
    """乱数固定"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


# Function to map an filter out control group
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


# Function to scale our data
def scaling(train, test, scaler=RobustScaler()):
    """規格化。pcaの後に実行してる。pcaの後だから外れ値にロバストな規格化使ってるみたい"""
    features = train.columns[2:]
    # scaler = RobustScaler()  # 外れ値に頑健な標準化
    scaler.fit(pd.concat([train[features], test[features]], axis=0))
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, features


# Function to extract pca features
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


# Function to extract common stats features
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


def g_squared(train, test):
    """gの特徴量について2乗した特徴量作成"""
    features_g = list(train.columns[4:776])
    for df in [train, test]:
        for feature in features_g:
            df[f"{feature}_squared"] = df[feature] ** 2
    return train, test


def c_squared(train, test):
    """cの特徴量について2乗した特徴量作成"""
    features_c = list(train.columns[776:876])
    for df in [train, test]:
        for feature in features_c:
            df[f"{feature}_squared"] = df[feature] ** 2
    return train, test


# Function to calculate the mean log loss of the targets including clipping
def mean_log_loss(y_true, y_pred):
    """マルチラベル全体でlog lossを平均する"""
    # 評価指標がlog losだからか？+label smoothingするため、予測ラベルはクリッピングする
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    metrics = []
    for target in range(206):
        metrics.append(log_loss(y_true[:, target], y_pred[:, target]))
    return np.mean(metrics)


def create_model_rs(shape1, shape2):
    """入力2つのNN.resnetみたくskip connection入れてる"""
    input_1 = tf.keras.layers.Input(shape=(shape1))
    input_2 = tf.keras.layers.Input(shape=(shape2))

    head_1 = tf.keras.layers.BatchNormalization()(input_1)
    head_1 = tf.keras.layers.Dropout(0.2)(head_1)
    head_1 = tf.keras.layers.Dense(512, activation="elu")(head_1)
    head_1 = tf.keras.layers.BatchNormalization()(head_1)
    input_3 = tf.keras.layers.Dense(256, activation="elu")(head_1)

    input_3_concat = tf.keras.layers.Concatenate()(
        [input_2, input_3]
    )  # node連結。node数が2つのnodeの足し算になる

    head_2 = tf.keras.layers.BatchNormalization()(input_3_concat)
    head_2 = tf.keras.layers.Dropout(0.3)(head_2)
    head_2 = tf.keras.layers.Dense(512, "relu")(head_2)
    head_2 = tf.keras.layers.BatchNormalization()(head_2)
    head_2 = tf.keras.layers.Dense(512, "elu")(head_2)
    head_2 = tf.keras.layers.BatchNormalization()(head_2)
    head_2 = tf.keras.layers.Dense(256, "relu")(head_2)
    head_2 = tf.keras.layers.BatchNormalization()(head_2)
    input_4 = tf.keras.layers.Dense(256, "elu")(head_2)

    input_4_avg = tf.keras.layers.Average()([input_3, input_4])  # 入力のリストを要素ごとに平均化

    head_3 = tf.keras.layers.BatchNormalization()(input_4_avg)
    head_3 = tf.keras.layers.Dense(
        256, kernel_initializer="lecun_normal", activation="selu"
    )(head_3)
    head_3 = tf.keras.layers.BatchNormalization()(head_3)
    head_3 = tf.keras.layers.Dense(
        206, kernel_initializer="lecun_normal", activation="selu"
    )(head_3)
    head_3 = tf.keras.layers.BatchNormalization()(head_3)
    output = tf.keras.layers.Dense(206, activation="sigmoid")(head_3)

    model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=output)
    opt = tf.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0015),  # ラベルスムージング
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/rs_shape.h5")
    return model


# Function to create our 5 layer dnn model
def create_model_5l(shape):
    """入力1つの5層NN。Stochastic Weight Averaging使う"""
    inp = tf.keras.layers.Input(shape=(shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(2560, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(2048, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1524, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1012, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(780, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(206, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = tf.optimizers.Adam(learning_rate=LR)
    opt = tfa.optimizers.SWA(
        opt
    )  # Stochastic Weight Averaging.モデルの重みを、これまで＋今回の平均を取って更新していくことでうまくいくみたい https://twitter.com/icoxfog417/status/989762534163992577
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0020),  # ラベルスムージング
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/5l_shape.h5")
    return model


# Function to create our 4 layer dnn model
def create_model_4l(shape, opt=tf.optimizers.Adam(learning_rate=LR)):
    """入力1つの4層NN。シンプル"""
    inp = tf.keras.layers.Input(shape=(shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(2048, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1524, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1012, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1012, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(206, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = tf.optimizers.Adam(learning_rate=LR)
    # opt = AdaBeliefOptimizer(learning_rate=LR, epsilon=1e-15)
    # print(opt)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0020),  # ラベルスムージング
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/4l_shape.h5")
    return model


# Function to create our 3 layer dnn model
def create_model_3l(shape):
    """入力1つの3層NN。Lookahead, WeightNormalization使う"""
    inp = tf.keras.layers.Input(shape=(shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.4914099166744246)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(1159, activation="relu"))(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.18817607797795838)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(960, activation="relu"))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.12542057776853896)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(1811, activation="relu"))(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20175242230280122)(x)
    out = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(206, activation="sigmoid")
    )(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = tf.optimizers.Adam(learning_rate=LR)
    opt = tfa.optimizers.Lookahead(
        opt, sync_period=10
    )  # Lookahead.勾配の更新方法を工夫してるみたい。学習率の設定にロバストが手法 https://cyberagent.ai/blog/research/11410/
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0015),
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/3l_shape.h5")
    return model


# Function to create our 2 layer dnn model
def create_model_2l(shape):
    """入力1つの2層NN。Lookahead, WeightNormalization使う"""
    inp = tf.keras.layers.Input(shape=(shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(0.2688628097505064)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(1292, activation="relu"))(
        x
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4598218403250696)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(983, activation="relu"))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4703144018483698)(x)
    out = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(206, activation="sigmoid")
    )(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = tf.optimizers.Adam(learning_rate=LR)
    opt = tfa.optimizers.Lookahead(opt, sync_period=10)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0015),
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/2l_shape.h5")
    return model


# Function to train our dnn
def train_and_evaluate(
    train,
    test,
    train_targets,
    features,
    start_predictors,
    SEED=123,
    MODEL="3l",
    PATH="../model/moa-3layer",
    OOF_TXT="tmp.txt",
    opt=tf.optimizers.Adam(learning_rate=LR),
):
    """モデル作成"""
    print(f"fold:{FOLDS}, epochs:{EPOCHS}")
    os.makedirs(PATH, exist_ok=True)
    seed_everything(SEED)
    oof_pred = np.zeros((train.shape[0], 206))
    test_pred = np.zeros((test.shape[0], 206))
    for fold, (trn_ind, val_ind) in tqdm(
        enumerate(
            MultilabelStratifiedKFold(
                n_splits=FOLDS, random_state=SEED, shuffle=True
            ).split(train_targets, train_targets)
        )
    ):
        model_path = f"{PATH}/{MODEL}_{fold}_{SEED}.h5"
        K.clear_session()
        if MODEL == "5l":
            model = create_model_5l(len(features))
        elif MODEL == "4l":
            model = create_model_4l(len(features), opt=opt)
        elif MODEL == "3l":
            model = create_model_3l(len(features))
        elif MODEL == "2l":
            model = create_model_2l(len(features))
        elif MODEL == "rs":
            model = create_model_rs(len(features), len(start_predictors))

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_binary_crossentropy",
            mode="min",
            patience=10,
            restore_best_weights=False,
            verbose=VERBOSE,
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_binary_crossentropy",
            mode="min",
            factor=0.3,
            patience=3,
            verbose=VERBOSE,
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_binary_crossentropy",
            verbose=VERBOSE,
            save_best_only=True,
            save_weights_only=True,
        )

        x_train, x_val = (
            train[features].values[trn_ind],
            train[features].values[val_ind],
        )
        y_train, y_val = train_targets.values[trn_ind], train_targets.values[val_ind]

        # ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int). 回避
        x_train = np.asarray(x_train).astype("float32")
        x_val = np.asarray(x_val).astype("float32")
        test[features] = np.asarray(test[features]).astype("float32")

        if MODEL == "rs":
            # 入力2つのNN使うから工夫してる
            x_train_, x_val_ = (
                train[start_predictors].values[trn_ind],
                train[start_predictors].values[val_ind],
            )

            model.fit(
                [x_train, x_train_],
                y_train,
                validation_data=([x_val, x_val_], y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stopping, reduce_lr, checkpoint],
                verbose=VERBOSE,
            )

            model.load_weights(model_path)

            oof_pred[val_ind] = model.predict([x_val, x_val_])
            test_pred += (
                model.predict([test[features].values, test[start_predictors].values])
                / FOLDS
            )

        else:
            model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stopping, reduce_lr, checkpoint],
                verbose=VERBOSE,
            )

            model.load_weights(model_path)

            oof_pred[val_ind] = model.predict(x_val)
            test_pred += model.predict(test[features].values) / FOLDS

    oof_score = mean_log_loss(train_targets.values, oof_pred)
    # str_loss = f"Our out of folds mean log loss score is {oof_score}"
    # print(str_loss)

    ## oofのlossの値残しておく
    # with open(OOF_TXT, mode="a") as f:
    #    f.write(f"{MODEL}_seed:{SEED}: {str_loss}\n")

    return test_pred, oof_pred


# Function to train our dnn
def inference(
    train,
    test,
    train_targets,
    features,
    start_predictors,
    SEED=123,
    MODEL="3l",
    PATH="../input/moa-3layer",
):
    """推論用"""
    seed_everything(SEED)
    oof_pred = np.zeros((train.shape[0], 206))
    test_pred = np.zeros((test.shape[0], 206))
    for fold, (trn_ind, val_ind) in enumerate(
        MultilabelStratifiedKFold(
            n_splits=FOLDS, random_state=SEED, shuffle=True
        ).split(train_targets, train_targets)
    ):
        K.clear_session()
        if MODEL == "5l":
            model = create_model_5l(len(features))
        elif MODEL == "4l":
            model = create_model_4l(len(features))
        elif MODEL == "3l":
            model = create_model_3l(len(features))
        elif MODEL == "2l":
            model = create_model_2l(len(features))
        elif MODEL == "rs":
            model = create_model_rs(len(features), len(start_predictors))

        x_train, x_val = (
            train[features].values[trn_ind],
            train[features].values[val_ind],
        )
        y_train, y_val = train_targets.values[trn_ind], train_targets.values[val_ind]

        model.load_weights(f"{PATH}/{MODEL}_{fold}_{SEED}.h5")

        if MODEL == "rs":
            x_train_, x_val_ = (
                train[start_predictors].values[trn_ind],
                train[start_predictors].values[val_ind],
            )
            oof_pred[val_ind] = model.predict([x_val, x_val_])
            test_pred += (
                model.predict([test[features].values, test[start_predictors].values])
                / FOLDS
            )
        else:
            oof_pred[val_ind] = model.predict(x_val)
            test_pred += model.predict(test[features].values) / FOLDS

    oof_score = mean_log_loss(train_targets.values, oof_pred)
    print(f"Our out of folds mean log loss score is {oof_score}")

    return test_pred, oof_pred


# Function to train our model with multiple seeds and average the predictions
def run_multiple_seeds(
    train,
    test,
    train_targets,
    features,
    start_predictors,
    SEEDS=[123],
    MODEL="3l",
    PATH="../input/moa-3layer",
):
    """cvのseed変えて推論をseed アンサンブル"""
    test_pred = []
    oof_pred = []

    for SEED in SEEDS:
        print(f"\nUsing model {MODEL} with seed {SEED} for inference")
        print(f"Trained with {len(features)} features")
        test_pred_, oof_pred_ = inference(
            train,
            test,
            train_targets,
            features,
            start_predictors,
            SEED=SEED,
            MODEL=MODEL,
            PATH=PATH,
        )
        test_pred.append(test_pred_)
        oof_pred.append(oof_pred_)
        print("-" * 50)
        print("\n")

    test_pred = np.average(test_pred, axis=0)
    oof_pred = np.average(oof_pred, axis=0)

    seed_log_loss = mean_log_loss(train_targets.values, oof_pred)
    print(f"Our out of folds log loss for our seed blend model is {seed_log_loss}")

    return test_pred, oof_pred


def submission(
    test_pred, test, sample_submission, train_targets, out_csv="submission.csv"
):
    sample_submission.loc[:, train_targets.columns] = test_pred
    sample_submission.loc[test["cp_type"] == 1, train_targets.columns] = 0
    sample_submission.to_csv(out_csv, index=False)
    return sample_submission


def run_train(model_dir="model"):
    """モデル作成"""
    os.makedirs(f"{model_dir}/moa-5layer", exist_ok=True)
    os.makedirs(f"{model_dir}/moa-4layer", exist_ok=True)
    os.makedirs(f"{model_dir}/moa-3layer", exist_ok=True)
    os.makedirs(f"{model_dir}/moa-2layer", exist_ok=True)
    os.makedirs(f"{model_dir}/moa-rs", exist_ok=True)

    # train time
    for seed1 in SEEDS1:
        test_pred_5l, oof_pred_5l = train_and_evaluate(
            train,
            test,
            train_targets,
            features,
            start_predictors,
            SEED=seed1,
            MODEL="5l",
            PATH=f"{model_dir}/moa-5layer",
        )
    for seed2 in SEEDS2:
        test_pred_4l, oof_pred_4l = train_and_evaluate(
            train,
            test,
            train_targets,
            features,
            start_predictors,
            SEED=seed2,
            MODEL="4l",
            PATH=f"{model_dir}/moa-4layer",
        )
    for seed3 in SEEDS3:
        test_pred_3l, oof_pred_3l = train_and_evaluate(
            train,
            test,
            train_targets,
            features,
            start_predictors,
            SEED=seed3,
            MODEL="3l",
            PATH=f"{model_dir}/moa-3layer",
        )
    for seed4 in SEEDS4:
        test_pred_2l, oof_pred_2l = train_and_evaluate(
            train,
            test,
            train_targets,
            features,
            start_predictors,
            SEED=seed4,
            MODEL="2l",
            PATH=f"{model_dir}/moa-2layer",
        )
    for seed5 in SEEDS5:
        test_pred_rs, oof_pred_rs = train_and_evaluate(
            train,
            test,
            train_targets,
            features,
            start_predictors,
            SEED=seed5,
            MODEL="rs",
            PATH=f"{model_dir}/moa-rs",
        )

    # Blend 5l, 4l, 3l and l2 dnn model
    oof_pred = np.average([oof_pred_5l, oof_pred_4l, oof_pred_3l, oof_pred_2l], axis=0)
    seed_log_loss = mean_log_loss(train_targets.values, oof_pred)
    str_loss1 = (
        f"Our final out of folds log loss for our classic dnn blend is {seed_log_loss}"
    )
    print(str_loss1)
    # test_pred = np.average([test_pred_5l, test_pred_4l, test_pred_3l, test_pred_2l], axis=0)

    # Blend the result of the previous model with the dnn resnet type model
    oof_pred = np.average([oof_pred, oof_pred_rs], axis=0)
    seed_log_loss = mean_log_loss(train_targets.values, oof_pred)
    str_loss2 = f"Our final out of folds log loss for our classic dnn + dnn resnet type model is {seed_log_loss}"
    print(str_loss2)
    # test_pred = np.average([test_pred, test_pred_rs], axis=0)

    # sample_submission = submission(test_pred)
    # sample_submission.head()

    # oofのlossの値残しておく
    with open(OOF_TXT, mode="a") as f:
        f.write(f"{str_loss1}\n{str_loss2}\n")


def run_submission(
    train, test, train_targets, features, sample_submission, model_dir="../input"
):
    """モデル推論してsubmissioファイル作成"""
    # Inference time
    test_pred_5l, oof_pred_5l = run_multiple_seeds(
        train,
        test,
        train_targets,
        features,
        start_predictors,
        SEEDS=SEEDS1,
        MODEL="5l",
        PATH=f"{model_dir}/moa-5layer",
    )
    test_pred_4l, oof_pred_4l = run_multiple_seeds(
        train,
        test,
        train_targets,
        features,
        start_predictors,
        SEEDS=SEEDS2,
        MODEL="4l",
        PATH=f"{model_dir}/moa-4layer",
    )
    test_pred_3l, oof_pred_3l = run_multiple_seeds(
        train,
        test,
        train_targets,
        features,
        start_predictors,
        SEEDS=SEEDS3,
        MODEL="3l",
        PATH=f"{model_dir}/moa-3layer",
    )
    test_pred_2l, oof_pred_2l = run_multiple_seeds(
        train,
        test,
        train_targets,
        features,
        start_predictors,
        SEEDS=SEEDS4,
        MODEL="2l",
        PATH=f"{model_dir}/moa-2layer",
    )
    test_pred_rs, oof_pred_rs = run_multiple_seeds(
        train,
        test,
        train_targets,
        features,
        start_predictors,
        SEEDS=SEEDS5,
        MODEL="rs",
        PATH=f"{model_dir}/moa-rs",
    )

    # Blend 5l, 4l, 3l and l2 dnn model
    oof_pred = np.average([oof_pred_5l, oof_pred_4l, oof_pred_3l, oof_pred_2l], axis=0)
    seed_log_loss = mean_log_loss(train_targets.values, oof_pred)
    print(
        f"Our final out of folds log loss for our classic dnn blend is {seed_log_loss}"
    )
    test_pred = np.average(
        [test_pred_5l, test_pred_4l, test_pred_3l, test_pred_2l], axis=0
    )

    # Blend the result of the previous model with the dnn resnet type model
    oof_pred = np.average([oof_pred, oof_pred_rs], axis=0)
    seed_log_loss = mean_log_loss(train_targets.values, oof_pred)
    print(
        f"Our final out of folds log loss for our classic dnn + dnn resnet type model is {seed_log_loss}"
    )
    test_pred = np.average([test_pred, test_pred_rs], axis=0)

    sample_submission = submission(test_pred, test, sample_submission, train_targets)
    # sample_submission.head()

    # ------- Nelder-Mead で最適なブレンディングの重み見つける -------
    oof_preds = [oof_pred_5l, oof_pred_4l, oof_pred_3l, oof_pred_2l, oof_pred_rs]
    best_weights = nelder_mead_weights(train_targets.values, oof_preds)
    oof_pred_weights = (
        best_weights[0] * oof_pred_5l
        + best_weights[1] * oof_pred_4l
        + best_weights[2] * oof_pred_3l
        + best_weights[3] * oof_pred_2l
        + best_weights[4] * oof_pred_rs
    )
    seed_log_loss = mean_log_loss(train_targets.values, oof_pred_weights)
    print(f"Nelder-Mead dnn blend is {seed_log_loss}")
    test_pred = (
        best_weights[0] * test_pred_5l
        + best_weights[1] * test_pred_4l
        + best_weights[2] * test_pred_3l
        + best_weights[3] * test_pred_2l
        + best_weights[4] * test_pred_rs
    )
    sample_submission = submission(test_pred, out_csv="submission_Nelder-Mead.csv")
    # ----------------------------------------------------------------


def nelder_mead_weights(y_true: np.ndarray, oof_preds: list):
    """ネルダーミードでモデルのブレンド重み最適化"""
    from scipy.optimize import minimize

    def opt(ws, y_true, y_preds):
        y_pred = None
        for w, y_p in zip(ws, y_preds):
            if y_pred is None:
                y_pred = w * y_p
            else:
                y_pred += w * y_p

        return mean_log_loss(y_true, y_pred)

    initial_weights = np.array([1.0 / len(oof_preds)] * len(oof_preds))
    result = minimize(
        opt, x0=initial_weights, args=(y_true, oof_preds), method="Nelder-Mead"
    )
    best_weights = result.x
    return best_weights


def run_train_1model(
    train,
    test,
    train_targets,
    features,
    logger=Logger("./"),
    model_dir="model",
    model_type="3l",
    seeds=[5],
    start_predictors=start_predictors,
):
    """1モデルだけ作成"""
    os.makedirs(f"{model_dir}/moa-{model_type}", exist_ok=True)
    try:
        # train time
        oof_pred = None
        for _seed in seeds:
            _test_pred, _oof_pred = train_and_evaluate(
                train,
                test,
                train_targets,
                features,
                start_predictors,
                SEED=_seed,
                MODEL=model_type,
                PATH=f"{model_dir}/moa-{model_type}",
            )
            if oof_pred is None:
                oof_pred = _oof_pred
            else:
                oof_pred += _oof_pred
        seed_log_loss = mean_log_loss(train_targets.values, oof_pred / len(seeds))

        # oofのlossの値残しておく
        seed_log_loss = round(seed_log_loss, 7)
        logger.result(
            f"model_type:{model_type}, seed:{str(seeds)}, oof:{str(seed_log_loss)}"
        )  # result.logに文字列書き込み

        return seed_log_loss
    except Exception as e:
        traceback.print_exc()
        # logger.result(f"Exception: {str(e)}")  # result.logに文字列書き込み
        return 0.0


if __name__ == "__main__":
    OOF_TXT = "oof.txt"
    OUT_MODEL = "model"

    train, train_targets, test, sample_submission = load_orig_data()

    train, train_targets, test = mapping_and_filter(train, train_targets, test)
    train, test = fe_stats(train, test)
    train, test = c_squared(train, test)
    train, test = fe_pca(train, test, n_components_g=70, n_components_c=10, SEED=123)
    train, test, features = scaling(train, test)

    # デバッグ用にデータ減らすか
    DEBUG = False
    # DEBUG = True
    if DEBUG:
        FOLDS = 3  # cvの数
        EPOCHS = 2

    with open(OOF_TXT, mode="w") as f:
        f.write(f"### start run_train: {datetime.datetime.now()} ###\n\n")

    run_train(model_dir=OUT_MODEL)

    with open(OOF_TXT, mode="a") as f:
        f.write(f"### end run_train: {datetime.datetime.now()} ###\n\n")

    run_submission(model_dir=OUT_MODEL)

    with open(OOF_TXT, mode="a") as f:
        f.write(f"### end run_submission: {datetime.datetime.now()} ###")
