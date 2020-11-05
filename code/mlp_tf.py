"""MLP tensorflow"""
import os
import gc
import pathlib
import sys
import traceback
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss, f1_score
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from datasets import drug_MultilabelStratifiedKFold
from params import (
    FOLDS,
    EPOCHS,
    BATCH_SIZE,
    LR,
    VERBOSE,
    DATADIR,
    # ITERATIVE_STRATIFICATION,
    ADABELIEF_TF,
)
from util import Logger
import tabnet_tf

# Adamを改良したAdaBelief https://github.com/juntang-zhuang/Adabelief-Optimizer
sys.path.append(ADABELIEF_TF)
from adabelief_tf import AdaBeliefOptimizer

# sys.path.append(ITERATIVE_STRATIFICATION)
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

warnings.filterwarnings("ignore")

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


def seed_everything(seed=123):
    """乱数固定"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


def submission_post_process(
    test_pred, out_csv="submission.csv", p_min=None, is_minor_correction=False,
):
    """submitファイル後処理+保存"""
    _train_targets = pd.read_csv(f"{DATADIR}/train_targets_scored.csv")
    _train_targets.drop(["sig_id"], inplace=True, axis=1)
    _test = pd.read_csv(f"{DATADIR}/test_features.csv")

    sample_submission = pd.read_csv(f"{DATADIR}/sample_submission.csv")
    sample_submission.loc[:, _train_targets.columns] = test_pred

    # 0,1になる予測値クリップするか
    if p_min is not None:
        # p_min = 0.001
        # p_max = 0.999
        sample_submission.iloc[:, 1:] = np.clip(
            sample_submission.iloc[:, 1:].values, p_min, 1.0 - p_min
        )

    # 正例が1件だけのクラス補正する
    # https://www.kaggle.com/c/lish-moa/discussion/191135
    if is_minor_correction:
        targets = ["atp-sensitive_potassium_channel_antagonist", "erbb2_inhibitor"]
        sample_submission.loc[:, targets] = 0.000012

    # crl行は0
    sample_submission.loc[_test["cp_type"] == 1, _train_targets.columns] = 0
    sample_submission.to_csv(out_csv, index=False)
    return sample_submission


# ===================================================================================================
# ---------------------------------------------- Loss -----------------------------------------------
# ===================================================================================================
def mean_log_loss(y_true, y_pred, n_class=206, p_min=1e-15):
    """マルチラベル全体でlog lossを平均する"""
    assert (
        y_true.shape[1] == n_class
    ), f"train_targetsの列数が {n_class} でないからlog_loss計算できない.y_true.shape: {y_true.shape}"

    # デフォルトでは 1e-15 でクリップするのは Evaluation がそうだから
    # https://www.kaggle.com/c/lish-moa/overview/evaluation
    y_pred = np.clip(y_pred, p_min, 1.0 - p_min)

    metrics = []
    for target in range(n_class):
        metrics.append(log_loss(y_true[:, target], y_pred[:, target]))
    return np.mean(metrics)


def mean_log_loss_train_targets_oof(y_pred, p_min=1e-15):
    """train_targetsのoof用mean_log_loss"""
    # mean_log_loss計算用に再ロード
    _train_targets = pd.read_csv(f"{DATADIR}/train_targets_scored.csv")
    _train_targets.drop(["sig_id"], inplace=True, axis=1)

    # oofのctl行の予測を0にする
    _train = pd.read_csv(f"{DATADIR}/train_features.csv")
    y_pred[_train["cp_type"] == 1, :] = 0

    return mean_log_loss(_train_targets.values, y_pred, n_class=206, p_min=p_min)


def get_oof_score(oof_pred, train_targets, train_targets_nonscored, p_min=1e-15):
    """oofのloss計算。列増えた時や行減った時での条件分岐あり"""
    # sig_id列残っているはずなので消す
    if "sig_id" in train_targets.columns:
        train_targets = train_targets.drop(["sig_id"], axis=1)

    if train_targets.shape[0] == 23814:
        oof_score = mean_log_loss_train_targets_oof(oof_pred, p_min=p_min)
    elif train_targets.shape[1] > 206:
        oof_score = mean_log_loss(
            train_targets.iloc[:, train_targets_nonscored.shape[1] :].values,
            oof_pred,
            p_min=1e-15,
        )
    else:
        oof_score = mean_log_loss(train_targets.values, oof_pred, p_min=1e-15)
    return oof_score


def clip_logloss(y_true, y_pred):
    """fit のmetrics でのクリップ用"""
    p_min = 0.0005
    p_max = 0.9995
    y_pred = tf.clip_by_value(y_pred, p_min, p_max)
    return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))


# ===================================================================================================
# ---------------------------------------------- Model ----------------------------------------------
# ===================================================================================================


def create_model_lr(shape, n_class=206, drop=0.5):
    """ロジスティック回帰.dropoutあり"""
    inp = tf.keras.layers.Input(shape=(shape))
    x = tf.keras.layers.Dropout(drop)(inp)
    out = tf.keras.layers.Dense(n_class, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.001),
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    return model


def create_model_3lWN(shape, n_class=206):
    """WeightNormalization使った3層のNN"""
    inp = tf.keras.layers.Input(shape=(shape))

    x = tf.keras.layers.BatchNormalization()(inp)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu"))(
        x
    )

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu"))(
        x
    )

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu"))(
        x
    )

    out = tf.keras.layers.Dense(n_class, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = tf.optimizers.Adam(learning_rate=LR, decay=1e-5)
    opt = tfa.optimizers.SWA(
        opt
    )  # Stochastic Weight Averaging.モデルの重みを、これまで＋今回の平均を取って更新していくことでうまくいくみたい https://twitter.com/icoxfog417/status/989762534163992577
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.001),  # ラベルスムージング
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/3lWN_shape.h5")
    return model


def create_model_rs(shape1, shape2, n_class=206):
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
        n_class, kernel_initializer="lecun_normal", activation="selu"
    )(head_3)
    head_3 = tf.keras.layers.BatchNormalization()(head_3)
    output = tf.keras.layers.Dense(n_class, activation="sigmoid")(head_3)

    model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=output)
    opt = tf.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0015),  # ラベルスムージング
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/rs_shape.h5")
    return model


def create_model_5l(shape, n_class=206):
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
    out = tf.keras.layers.Dense(n_class, activation="sigmoid")(x)
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


def create_model_4l(shape, n_class=206):
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
    out = tf.keras.layers.Dense(n_class, activation="sigmoid")(x)
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


def create_model_3l(shape, n_class=206):
    """入力1つの3層NN。Lookahead, WeightNormalization 使う"""
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
        tf.keras.layers.Dense(n_class, activation="sigmoid")
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


def create_model_3l_v2(shape, n_class=206):
    """入力1つの3層NN。WeightNormalization adabelief_tf 使う"""
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
        tf.keras.layers.Dense(n_class, activation="sigmoid")
    )(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = AdaBeliefOptimizer(learning_rate=LR)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0015),
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/3l_shape.h5")
    return model


def create_model_2l(shape, n_class=206):
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
        tf.keras.layers.Dense(n_class, activation="sigmoid")
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


def create_model_stack_tabnet(
    shape, n_class=206, params=None, is_SWA=False, is_Lookahead=False, is_adabelief=True
):
    #    params = dict(
    #        feature_columns=None,
    #        num_classes=n_class,
    #        # num_layers=2,
    #        #        feature_dim=32,
    #        #        output_dim=32,
    #        num_features=shape,
    #        # num_decision_steps=2,
    #        #        relaxation_factor=1.3,
    #        #        sparsity_coefficient=0.0,
    #        #        batch_momentum=0.98,
    #        #        virtual_batch_size=24,
    #        #        norm_type="group",
    #        num_groups=-1,
    #        multi_label=True,
    #    )
    print("params:", params)
    if params is None:
        model = tabnet_tf.StackedTabNetClassifier(
            feature_columns=None,
            num_classes=n_class,
            num_layers=2,
            feature_dim=128,
            output_dim=64,
            num_features=shape,
            num_decision_steps=1,
            relaxation_factor=1.5,
            sparsity_coefficient=1e-5,
            batch_momentum=0.98,
            virtual_batch_size=None,
            norm_type="group",
            num_groups=-1,
            multi_label=True,
        )
    else:
        model = tabnet_tf.StackedTabNetClassifier(**params)
    if is_adabelief:
        opt = AdaBeliefOptimizer(learning_rate=LR)
    else:
        opt = tf.optimizers.Adam(LR)
    if is_SWA:
        opt = tfa.optimizers.Lookahead(opt, sync_period=10)
    if is_Lookahead:
        opt = tfa.optimizers.SWA(opt)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.001),
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    return model


def create_model_tabnet_class(
    shape, n_class=206, params=None, is_SWA=False, is_Lookahead=False, is_adabelief=True
):
    print("params:", params)
    if params is None:
        model = tabnet_tf.TabNetClassifier(
            feature_columns=None,
            num_classes=n_class,
            feature_dim=128,
            output_dim=64,
            num_features=shape,
            num_decision_steps=1,
            relaxation_factor=1.5,
            sparsity_coefficient=1e-5,
            batch_momentum=0.98,
            virtual_batch_size=None,
            norm_type="group",
            num_groups=-1,
            multi_label=True,
        )
    else:
        model = tabnet_tf.TabNetClassifier(**params)
    if is_adabelief:
        opt = AdaBeliefOptimizer(learning_rate=LR)
    else:
        opt = tf.optimizers.Adam(LR)
    if is_SWA:
        opt = tfa.optimizers.Lookahead(opt, sync_period=10)
    if is_Lookahead:
        opt = tfa.optimizers.SWA(opt)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.001),
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    return model


# ===================================================================================================
# ------------------------------------------ Train/Prewdict -----------------------------------------
# ===================================================================================================


def train_and_evaluate(
    train,
    test,
    train_targets,  # sig_id列必要
    features,
    train_targets_nonscored,  # sig_id列消しておくこと
    start_predictors=start_predictors,
    seeds=[123],
    model_type="3l",
    model_dir="mlp_tf",
    tabnet_params=None,
):
    """モデル作成"""
    print(f"fold:{FOLDS}, epochs:{EPOCHS}, batch_size:{BATCH_SIZE}, LR:{LR}")
    os.makedirs(model_dir, exist_ok=True)

    test_pred_seed = []
    oof_pred_seed = []

    for seed in seeds:
        seed_everything(seed)

        print(f"Using model {model_type} with seed {seed} for train_and_evaluate")
        print(
            f"Trained with {len(features)} features. train.shape: {train.shape}. train_targets.shape: {train_targets.shape}"
        )

        oof_pred = np.zeros((train.shape[0], 206))
        test_pred = np.zeros((test.shape[0], 206))

        # for fold, (trn_ind, val_ind) in tqdm(
        #    enumerate(
        #        MultilabelStratifiedKFold(
        #            n_splits=FOLDS, random_state=seed, shuffle=True
        #        ).split(train_targets, train_targets)
        #    )
        # ):
        # # MultiLabelStratifiedKFold(n_splits=5, shuffle=False) で乱数固定する 20201028
        # for fold, (trn_ind, val_ind) in tqdm(enumerate(MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=False).split(train_targets, train_targets))):
        # 薬物およびマルチラベル層別化 20201104
        # scored = drug_MultilabelStratifiedKFold(seed=seed, scored=train_targets,)  # MultilabelStratifiedKFoldの乱数指定
        scored = drug_MultilabelStratifiedKFold(
            scored=train_targets,
        )  # MultilabelStratifiedKFoldの乱数固定
        for fold in tqdm(range(FOLDS)):
            val_ind = scored[scored["fold"] == fold].index
            trn_ind = scored[scored["fold"] != fold].index

            # sig_id列残っているはずなので消す
            if "sig_id" in train_targets.columns:
                train_targets = train_targets.drop(["sig_id"], axis=1)
                # train_targets.drop(["sig_id"], inplace=True, axis=1)

            K.clear_session()
            if model_type == "5l":
                model = create_model_5l(len(features), n_class=train_targets.shape[1])
            elif model_type == "4l":
                model = create_model_4l(len(features), n_class=train_targets.shape[1])
            elif model_type == "3l":
                model = create_model_3l(len(features), n_class=train_targets.shape[1])
            elif model_type == "3l_v2":
                model = create_model_3l_v2(
                    len(features), n_class=train_targets.shape[1]
                )
            elif model_type == "2l":
                model = create_model_2l(len(features), n_class=train_targets.shape[1])
            elif model_type == "rs":
                model = create_model_rs(
                    len(features), len(start_predictors), n_class=train_targets.shape[1]
                )
            elif model_type == "3lWN":
                model = create_model_3lWN(len(features), n_class=train_targets.shape[1])
            elif model_type == "lr":
                model = create_model_lr(len(features), n_class=train_targets.shape[1])
            elif model_type == "stack_tabnet":
                model = create_model_stack_tabnet(
                    len(features), n_class=train_targets.shape[1], params=tabnet_params,
                )
            elif model_type == "tabnet_class":
                model = create_model_tabnet_class(
                    len(features), n_class=train_targets.shape[1], params=tabnet_params,
                )
            model_path = f"{model_dir}/{model_type}_{fold}_{seed}.h5"

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
            y_train, y_val = (
                train_targets.values[trn_ind],
                train_targets.values[val_ind],
            )

            # ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int). 回避
            x_train = np.asarray(x_train).astype("float32")
            x_val = np.asarray(x_val).astype("float32")
            test[features] = test[features].astype("float32")

            if model_type == "rs":
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

                y_pred_val = model.predict([x_val, x_val_])
                y_pred_test = (
                    model.predict(
                        [test[features].values, test[start_predictors].values]
                    )
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

                y_pred_val = model.predict(x_val)
                y_pred_test = model.predict(test[features].values) / FOLDS

            if y_pred_val.shape[1] > 206:
                # targetの列だけにする
                y_pred_val = y_pred_val[:, train_targets_nonscored.shape[1] :]
                y_pred_test = y_pred_test[:, train_targets_nonscored.shape[1] :]
                assert (
                    y_pred_val.shape[1] == 206
                ), f"予測値が206列でないのでnonscored残ってる。shape:{y_pred_val.shape[1]} 。"

            oof_pred[val_ind] = y_pred_val
            test_pred += y_pred_test

            del model
            gc.collect()

        oof_score = get_oof_score(oof_pred, train_targets, train_targets_nonscored)
        print(f"Our out of folds mean log loss score is {oof_score}")

        test_pred_seed.append(test_pred)
        oof_pred_seed.append(oof_pred)

    test_pred_avg = np.average(test_pred_seed, axis=0)
    oof_pred_avg = np.average(oof_pred_seed, axis=0)

    seed_log_loss = get_oof_score(oof_pred_avg, train_targets, train_targets_nonscored)
    print(f"Our out of folds log loss for our seed blend model is {seed_log_loss}")

    sample_submission_pred = submission_post_process(
        test_pred_avg, out_csv=f"{model_dir}/submission.csv"
    )

    return test_pred_avg, oof_pred_avg


def inference(
    train,
    test,
    train_targets,  # sig_id列必要
    features,
    train_targets_nonscored,  # sig_id列消しておくこと
    start_predictors=start_predictors,
    seeds=[123],
    model_type="3l",
    model_dir="mlp_tf",
    tabnet_params=None,
):
    """推論用"""
    test_pred_seed = []
    oof_pred_seed = []

    for seed in seeds:
        seed_everything(seed)

        print(f"Using model {model_type} with seed {seed} for inference")
        print(f"Trained with {len(features)} features")

        oof_pred = np.zeros((train.shape[0], 206))
        test_pred = np.zeros((test.shape[0], 206))

        # for fold, (trn_ind, val_ind) in enumerate(
        #    MultilabelStratifiedKFold(
        #        n_splits=FOLDS, random_state=seed, shuffle=True
        #    ).split(train_targets, train_targets)
        # ):
        # # MultiLabelStratifiedKFold(n_splits=5, shuffle=False) で乱数固定する 20201028
        # for fold, (trn_ind, val_ind) in tqdm(enumerate(MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=False).split(train_targets, train_targets))):
        # 薬物およびマルチラベル層別化 20201104
        # scored = drug_MultilabelStratifiedKFold(seed=seed, scored=train_targets,)  # MultilabelStratifiedKFoldの乱数指定
        scored = drug_MultilabelStratifiedKFold(
            scored=train_targets
        )  # MultilabelStratifiedKFoldの乱数固定
        for fold in tqdm(range(FOLDS)):
            val_ind = scored[scored["fold"] == fold].index
            trn_ind = scored[scored["fold"] != fold].index

            # sig_id列残っているはずなので消す
            if "sig_id" in train_targets.columns:
                train_targets = train_targets.drop(["sig_id"], axis=1)
                # train_targets.drop(["sig_id"], inplace=True, axis=1)

            K.clear_session()
            x_train, x_val = (
                train[features].values[trn_ind],
                train[features].values[val_ind],
            )
            y_train, y_val = (
                train_targets.values[trn_ind],
                train_targets.values[val_ind],
            )

            if model_type == "5l":
                model = create_model_5l(len(features), n_class=train_targets.shape[1])
            elif model_type == "4l":
                model = create_model_4l(len(features), n_class=train_targets.shape[1])
            elif model_type == "3l":
                model = create_model_3l(len(features), n_class=train_targets.shape[1])
            elif model_type == "3l_v2":
                model = create_model_3l_v2(
                    len(features), n_class=train_targets.shape[1]
                )
            elif model_type == "2l":
                model = create_model_2l(len(features), n_class=train_targets.shape[1])
            elif model_type == "rs":
                model = create_model_rs(
                    len(features), len(start_predictors), n_class=train_targets.shape[1]
                )
            elif model_type == "3lWN":
                model = create_model_3lWN(len(features), n_class=train_targets.shape[1])
            elif model_type == "lr":
                model = create_model_lr(len(features), n_class=train_targets.shape[1])
            elif model_type == "stack_tabnet":
                model = create_model_stack_tabnet(
                    len(features), n_class=train_targets.shape[1], params=tabnet_params,
                )
                # tabnetはサブクラスモデルなのでfit しないとロードできないため、1エポックだけ実行
                model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=1,
                    batch_size=BATCH_SIZE,
                    verbose=VERBOSE,
                )
            elif model_type == "tabnet_class":
                model = create_model_tabnet_class(
                    len(features), n_class=train_targets.shape[1], params=tabnet_params,
                )
                # tabnetはサブクラスモデルなのでfit しないとロードできないため、1エポックだけ実行
                model.fit(
                    x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=1,
                    batch_size=BATCH_SIZE,
                    verbose=VERBOSE,
                )
            model.load_weights(f"{model_dir}/{model_type}_{fold}_{seed}.h5")

            if model_type == "rs":
                x_train_, x_val_ = (
                    train[start_predictors].values[trn_ind],
                    train[start_predictors].values[val_ind],
                )
                y_pred_val = model.predict([x_val, x_val_])
                y_pred_test = (
                    model.predict(
                        [test[features].values, test[start_predictors].values]
                    )
                    / FOLDS
                )
            else:
                y_pred_val = model.predict(x_val)
                y_pred_test = model.predict(test[features].values) / FOLDS

            if y_pred_val.shape[1] > 206:
                # targetの列だけにする
                y_pred_val = y_pred_val[:, train_targets_nonscored.shape[1] :]
                y_pred_test = y_pred_test[:, train_targets_nonscored.shape[1] :]
                assert (
                    y_pred_val.shape[1] == 206
                ), f"予測値が206列でないのでnonscored残ってる。shape:{y_pred_val.shape[1]} 。"

            oof_pred[val_ind] = y_pred_val
            test_pred += y_pred_test

            del model
            gc.collect()

        oof_score = get_oof_score(oof_pred, train_targets, train_targets_nonscored)
        print(f"Our out of folds mean log loss score is {oof_score}")

        test_pred_seed.append(test_pred)
        oof_pred_seed.append(oof_pred)

    test_pred_avg = np.average(test_pred_seed, axis=0)
    oof_pred_avg = np.average(oof_pred_seed, axis=0)

    seed_log_loss = get_oof_score(oof_pred_avg, train_targets, train_targets_nonscored)
    print(f"Our out of folds log loss for our seed blend model is {seed_log_loss}")

    sample_submission_pred = submission_post_process(
        test_pred_avg, out_csv=f"submission.csv"
    )

    return test_pred_avg, oof_pred_avg


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


# ===================================================================================================
# ---------------------------------------------- Logging --------------------------------------------
# ===================================================================================================


def run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    logger=Logger("./"),
    model_dir="mlp_tf",
    model_type="3l",
    seeds=[123],
    str_condition="",
    p_min=1e-15,
    is_train=True,
    tabnet_params=None,
):
    """モデル作成/推論してログファイルに結果書き込む"""
    str_train_flag = "train" if is_train else "inference"
    if is_train:
        # train
        test_pred, oof_pred = train_and_evaluate(
            train,
            test,
            train_targets,
            features,
            train_targets_nonscored,
            seeds=seeds,
            model_type=model_type,
            model_dir=model_dir,
            tabnet_params=tabnet_params,
        )
    else:
        # inference
        test_pred, oof_pred = inference(
            train,
            test,
            train_targets,
            features,
            train_targets_nonscored,
            seeds=seeds,
            model_type=model_type,
            model_dir=model_dir,
            tabnet_params=tabnet_params,
        )
    oof_log_loss = get_oof_score(
        oof_pred, train_targets, train_targets_nonscored, p_min=p_min
    )
    oof_log_loss = round(oof_log_loss, 7)
    logger.info(
        f"model_type:{model_type}, oof:{str(oof_log_loss)}, train_flag:{str_train_flag}"
    )  # general.logに文字列書き込み
    logger.result(
        f"{model_type}\t{str_condition}\t{str(oof_log_loss)}\t{str_train_flag}"
    )  # result.logに文字列書き込み
    return test_pred, oof_pred


def run_mlp_tf_blend_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    logger=Logger("./"),
    model_dir="mlp_tf",
    seeds=[123],
    str_condition="",
    model_types=[
        "lr",
        "2l",
        "3l",
        "4l",
        "5l",
        "rs",
        "3l_v2",
        "3lWN",
        "stack_tabnet",
        "tabnet_class",
    ],
    is_train=True,
    is_nelder=False,
):
    """モデルブレンディングしてログファイルに結果書き込む"""
    str_train_flag = "train" if is_train else "inference"
    test_preds = []
    oof_preds = []

    def _run_model(_model_type, _str_condition):
        test_pred, oof_pred = run_mlp_tf_logger(
            train,
            test,
            train_targets,
            features,
            train_targets_nonscored,
            logger=logger,
            model_dir=model_dir,
            seeds=seeds,
            model_type=_model_type,
            str_condition=_str_condition,
            is_train=is_train,
        )
        test_preds.append(test_pred)
        oof_preds.append(oof_pred)

    for m_t in model_types:
        _run_model(m_t, str_condition)
    model_type = "-".join(model_types)
    mean_oof = np.average(oof_preds, axis=0)
    mean_oof = mean_oof / len(oof_preds)
    log_loss = get_oof_score(mean_oof, train_targets, train_targets_nonscored)
    log_loss = round(log_loss, 7)
    logger.info(
        f"model_type:{model_type}, oof:{str(log_loss)}, train_flag:{str_train_flag}"
    )  # general.logに文字列書き込み
    logger.result(
        f"{model_type}\t{str_condition}:Mean blend\t{str(log_loss)}\t{str_train_flag}"
    )  # result.logに文字列書き込み
    mean_test = np.average(test_preds, axis=0)
    mean_test = mean_test / len(test_preds)
    os.makedirs(f"mean", exist_ok=True)
    sample_submission_pred = submission_post_process(
        mean_test, out_csv=f"mean/submission.csv"
    )
    if is_nelder:
        # 時間かかるから基本False
        # ------- Nelder-Mead で最適なブレンディングの重み見つける -------
        print("running Nelder-Mead...")
        _train_targets = pd.read_csv(f"{DATADIR}/train_targets_scored.csv")
        _train_targets.drop(["sig_id"], inplace=True, axis=1)
        best_weights = nelder_mead_weights(_train_targets.values, oof_preds)
        wei_oof = None
        for wei, pre in zip(best_weights, oof_preds):
            if wei_oof is None:
                wei_oof = wei * pre
            else:
                wei_oof += wei * pre
        log_loss = get_oof_score(wei_oof, train_targets, train_targets_nonscored)
        log_loss = round(log_loss, 7)
        logger.info(
            f"model_type:{model_type}, oof:{str(log_loss)}, train_flag:{str_train_flag}"
        )  # general.logに文字列書き込み
        logger.result(
            f"{model_type}\t{str_condition}:Nelder-Mead blend\t{str(log_loss)}\t{str_train_flag}"
        )  # result.logに文字列書き込み
        wei_test = None
        for wei, pre in zip(best_weights, test_preds):
            if wei_test is None:
                wei_test = wei * pre
            else:
                wei_test += wei * pre
        os.makedirs(f"nelder", exist_ok=True)
        sample_submission_pred = submission_post_process(
            wei_test, out_csv=f"nelder/submission.csv"
        )
        # ----------------------------------------------------------------
