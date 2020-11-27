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

# 参考: https://www.kaggle.com/yxohrxn/mlpclassifier-fit?scriptVersionId=46905918

# + execution={"iopub.execute_input": "2020-11-15T04:40:11.867203Z", "iopub.status.busy": "2020-11-15T04:40:11.866191Z", "iopub.status.idle": "2020-11-15T04:40:11.869103Z", "shell.execute_reply": "2020-11-15T04:40:11.868399Z"} papermill={"duration": 0.029988, "end_time": "2020-11-15T04:40:11.869223", "exception": false, "start_time": "2020-11-15T04:40:11.839235", "status": "completed"} tags=[]
import sys

# sys.path.append("../input/adabeliefoptimizer/pypi_packages/adabelief_tf0.1.0")

# + execution={"iopub.execute_input": "2020-11-15T04:40:11.916751Z", "iopub.status.busy": "2020-11-15T04:40:11.915736Z", "iopub.status.idle": "2020-11-15T04:40:11.919098Z", "shell.execute_reply": "2020-11-15T04:40:11.918390Z"} papermill={"duration": 0.028727, "end_time": "2020-11-15T04:40:11.919213", "exception": false, "start_time": "2020-11-15T04:40:11.890486", "status": "completed"} tags=[]
import sys

# sys.path.append("../input/iterative-stratification/iterative-stratification-master")

# +
## GPU使わない
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# tensorflow2.0 + kerasでGPUメモリの使用量を抑える方法(最小限だけ使うように設定)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" execution={"iopub.execute_input": "2020-11-15T04:40:11.967087Z", "iopub.status.busy": "2020-11-15T04:40:11.966155Z", "iopub.status.idle": "2020-11-15T04:40:17.400927Z", "shell.execute_reply": "2020-11-15T04:40:17.400141Z"} papermill={"duration": 5.46079, "end_time": "2020-11-15T04:40:17.401049", "exception": false, "start_time": "2020-11-15T04:40:11.940259", "status": "completed"} tags=[]
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm.notebook import tqdm

# + code_folding=[3, 5] execution={"iopub.execute_input": "2020-11-15T04:40:17.455544Z", "iopub.status.busy": "2020-11-15T04:40:17.454466Z", "iopub.status.idle": "2020-11-15T04:40:17.457712Z", "shell.execute_reply": "2020-11-15T04:40:17.457108Z"} papermill={"duration": 0.034006, "end_time": "2020-11-15T04:40:17.457833", "exception": false, "start_time": "2020-11-15T04:40:17.423827", "status": "completed"} tags=[]
import tensorflow as tf


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


# + code_folding=[5, 56, 91, 120, 154]
import tensorflow as tf
import tensorflow_addons as tfa
from adabelief_tf import AdaBeliefOptimizer


def create_model_rs(shape1, shape2, num_classes=206):
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
        num_classes, kernel_initializer="lecun_normal", activation="selu"
    )(head_3)
    head_3 = tf.keras.layers.BatchNormalization()(head_3)
    output = tf.keras.layers.Dense(num_classes, activation="sigmoid")(head_3)

    model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=output)
    # rs_lr = 0.03  # C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\notebook\base_line\20201105
    # print(f"rs_lr: {rs_lr}")
    opt = tf.optimizers.Adam(learning_rate=params["lr"])
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0015),  # ラベルスムージング
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/rs_shape.h5")
    return model


def create_model_rs_v2(
    shape1,
    shape2,
    num_classes=206,
    activations=["elu", "elu", "relu", "elu", "relu", "elu", "selu", "selu"],
    denses=[512, 256, 512, 512, 256, 256],
    drop_rates=[0.2, 0.3],
):
    """入力2つのNN.resnetみたくskip connection入れてる"""
    input_1 = tf.keras.layers.Input(shape=(shape1))
    input_2 = tf.keras.layers.Input(shape=(shape2))

    head_1 = tf.keras.layers.BatchNormalization()(input_1)
    head_1 = tf.keras.layers.Dropout(drop_rates[0])(head_1)
    head_1 = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[0], activation=activations[0])
    )(head_1)
    head_1 = tf.keras.layers.BatchNormalization()(head_1)
    input_3 = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[1], activation=activations[1])
    )(head_1)

    input_3_concat = tf.keras.layers.Concatenate()(
        [input_2, input_3]
    )  # node連結。node数が2つのnodeの足し算になる

    head_2 = tf.keras.layers.BatchNormalization()(input_3_concat)
    head_2 = tf.keras.layers.Dropout(drop_rates[1])(head_2)
    head_2 = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[2], activations[2])
    )(head_2)
    head_2 = tf.keras.layers.BatchNormalization()(head_2)
    head_2 = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[3], activations[3])
    )(head_2)
    head_2 = tf.keras.layers.BatchNormalization()(head_2)
    head_2 = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[4], activations[4])
    )(head_2)
    head_2 = tf.keras.layers.BatchNormalization()(head_2)
    input_4 = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[1], activations[5])
    )(
        head_2
    )  # input_3 と同じnode数でないとだめ. tf.keras.layers.Average するから

    input_4_avg = tf.keras.layers.Average()([input_3, input_4])  # 入力のリストを要素ごとに平均化

    head_3 = tf.keras.layers.BatchNormalization()(input_4_avg)
    head_3 = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(
            denses[5], kernel_initializer="lecun_normal", activation=activations[6]
        )
    )(head_3)
    head_3 = tf.keras.layers.BatchNormalization()(head_3)
    head_3 = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(
            num_classes, kernel_initializer="lecun_normal", activation=activations[7]
        )
    )(head_3)
    head_3 = tf.keras.layers.BatchNormalization()(head_3)
    output = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(num_classes, activation="sigmoid")
    )(head_3)

    model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=output)
    # opt = tf.optimizers.Adam(learning_rate=params["lr"])
    opt = AdaBeliefOptimizer(learning_rate=params["lr"])
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0015),  # ラベルスムージング
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/rs_shape.h5")
    return model


def create_model_5l(
    shape,
    num_classes=206,
    activation="relu",
    denses=[2560, 2048, 1524, 1012, 780],
    drop_rates=[0.4, 0.4, 0.4, 0.4, 0.4, 0.2],
):
    """入力1つの5層NN。Stochastic Weight Averaging使う"""
    inp = tf.keras.layers.Input(shape=(shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(drop_rates[0])(x)
    x = tf.keras.layers.Dense(denses[0], activation=activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[1])(x)
    x = tf.keras.layers.Dense(denses[1], activation=activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[2])(x)
    x = tf.keras.layers.Dense(denses[2], activation=activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[3])(x)
    x = tf.keras.layers.Dense(denses[3], activation=activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[4])(x)
    x = tf.keras.layers.Dense(denses[4], activation=activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[5])(x)
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = tf.optimizers.Adam(learning_rate=params["lr"])
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


def create_model_5l_v2(
    shape,
    num_classes=206,
    activation="relu",
    denses=[2560, 2048, 1524, 1012, 780],
    drop_rates=[0.4, 0.4, 0.4, 0.4, 0.4, 0.2],
):
    """入力1つの5層NN。Stochastic Weight Averaging使う"""
    inp = tf.keras.layers.Input(shape=(shape))

    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(drop_rates[0])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[0], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[1])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[1], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[2])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[2], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[3])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[3], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[4])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[4], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[5])(x)
    out = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(num_classes, activation="sigmoid")
    )(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = tf.optimizers.Adam(learning_rate=params["lr"])
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


def create_model_4l(shape, num_classes=206):
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
    out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = tf.optimizers.Adam(learning_rate=params["lr"])
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0020),  # ラベルスムージング
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/4l_shape.h5")
    return model


def create_model_4l_v2(
    shape,
    num_classes=206,
    activation="relu",
    denses=[2048, 1524, 1012, 1012],
    drop_rates=[0.4, 0.4, 0.4, 0.4, 0.2],
):
    """入力1つの4層NN。シンプル"""
    inp = tf.keras.layers.Input(shape=(shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(drop_rates[0])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[0], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[1])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[1], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[2])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[2], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[3])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[3], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[4])(x)
    out = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(num_classes, activation="sigmoid")
    )(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = tf.optimizers.Adam(learning_rate=params["lr"])
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0020),  # ラベルスムージング
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/4l_shape.h5")
    return model


def create_model_3l_v3(
    shape,
    num_classes=206,
    activation="relu",
    denses=[1159, 960, 1811],
    drop_rates=[
        0.4914099166744246,
        0.18817607797795838,
        0.12542057776853896,
        0.20175242230280122,
    ],
):
    """入力1つの3層NN。WeightNormalization adabelief_tf 使う"""
    inp = tf.keras.layers.Input(shape=(shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(drop_rates[0])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[0], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[1])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[1], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[2])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[2], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[3])(x)
    out = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(num_classes, activation="sigmoid")
    )(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = AdaBeliefOptimizer(learning_rate=params["lr"])
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
    # _lr = 0.01  # C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\notebook\base_line\20201105
    # print(f"3l_v2_lr: {_lr}")
    opt = AdaBeliefOptimizer(learning_rate=_lr)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0015),
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/3l_shape.h5")
    return model


def create_model_2l_v2(
    shape,
    num_classes=206,
    activation="relu",
    denses=[1292, 983],
    drop_rates=[0.2688628097505064, 0.4598218403250696, 0.4703144018483698],
    sync_period=10,
):
    """入力1つの2層NN。Lookahead, WeightNormalization使う"""
    inp = tf.keras.layers.Input(shape=(shape))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(drop_rates[0])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[0], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[1])(x)
    x = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(denses[1], activation=activation)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(drop_rates[2])(x)
    out = tfa.layers.WeightNormalization(
        tf.keras.layers.Dense(num_classes, activation="sigmoid")
    )(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    opt = tf.optimizers.Adam(learning_rate=params["lr"])
    opt = tfa.optimizers.Lookahead(opt, sync_period=sync_period)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0015),
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/2l_shape.h5")
    return model


def create_model_2l(shape, num_classes=206):
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
        tf.keras.layers.Dense(num_classes, activation="sigmoid")
    )(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    # _lr = 0.1  # C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\notebook\base_line\20201105
    # print(f"2l_lr: {_lr}")
    opt = tf.optimizers.Adam(learning_rate=params["lr"])
    opt = tfa.optimizers.Lookahead(opt, sync_period=10)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0015),
        metrics=tf.keras.metrics.BinaryCrossentropy(),
    )
    # model.save("model/2l_shape.h5")
    return model


# + code_folding=[10]
import sys

sys.path.append(r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\code")
from tabnet_tf import *

# from tabnet import StackedTabNet

import tensorflow as tf
from adabelief_tf import AdaBeliefOptimizer


class StackedTabNetClassifier(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        batch_momentum=0.98,
        epsilon=1e-05,
        feature_columns=None,
        feature_dim=64,
        norm_type="group",
        num_decision_steps=5,
        num_features=None,
        num_groups=2,
        num_layers=1,
        output_dim=64,
        relaxation_factor=1.5,
        sparsity_coefficient=1e-05,
        virtual_batch_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.stacked_tabnet = StackedTabNet(
            feature_columns,
            batch_momentum=batch_momentum,
            epsilon=epsilon,
            feature_dim=feature_dim,
            norm_type=norm_type,
            num_decision_steps=num_decision_steps,
            num_features=num_features,
            num_groups=num_groups,
            num_layers=num_layers,
            output_dim=output_dim,
            relaxation_factor=relaxation_factor,
            sparsity_coefficient=sparsity_coefficient,
            virtual_batch_size=virtual_batch_size,
        )

        self.classifier = tf.keras.layers.Dense(
            num_classes, activation="sigmoid", use_bias=False
        )

    def call(self, inputs, training=None):
        x = self.stacked_tabnet(inputs, training=training)

        return self.classifier(x)


def create_model_stacked_tabnet_v2(n_features, stacked_tabnet_params, num_classes=206):
    # hyperparameters
    label_smoothing = 1e-03
    lr = 0.001
    model = StackedTabNetClassifier(
        num_classes=num_classes, num_features=n_features, **stacked_tabnet_params,
    )
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    optimizer = AdaBeliefOptimizer(learning_rate=lr)
    model.compile(loss=loss, optimizer=optimizer)
    return model


# + execution={"iopub.execute_input": "2020-11-15T04:40:17.701600Z", "iopub.status.busy": "2020-11-15T04:40:17.700641Z", "iopub.status.idle": "2020-11-15T04:40:17.703680Z", "shell.execute_reply": "2020-11-15T04:40:17.702958Z"} _kg_hide-input=true papermill={"duration": 0.029291, "end_time": "2020-11-15T04:40:17.703796", "exception": false, "start_time": "2020-11-15T04:40:17.674505", "status": "completed"} tags=[]
# def compute_absolute_features(X):
#     X = X.abs()

#     return X.rename(columns="{}_abs".format)

# + execution={"iopub.execute_input": "2020-11-15T04:40:17.753230Z", "iopub.status.busy": "2020-11-15T04:40:17.752132Z", "iopub.status.idle": "2020-11-15T04:40:17.755556Z", "shell.execute_reply": "2020-11-15T04:40:17.754829Z"} _kg_hide-input=true papermill={"duration": 0.02971, "end_time": "2020-11-15T04:40:17.755681", "exception": false, "start_time": "2020-11-15T04:40:17.725971", "status": "completed"} tags=[]
# def compute_square_features(X):
#     X = X ** 2

#     return X.rename(columns="{}_square".format)

# + execution={"iopub.execute_input": "2020-11-15T04:40:17.805648Z", "iopub.status.busy": "2020-11-15T04:40:17.804620Z", "iopub.status.idle": "2020-11-15T04:40:17.807511Z", "shell.execute_reply": "2020-11-15T04:40:17.808039Z"} _kg_hide-input=true papermill={"duration": 0.030132, "end_time": "2020-11-15T04:40:17.808189", "exception": false, "start_time": "2020-11-15T04:40:17.778057", "status": "completed"} tags=[]
# import pandas as pd


# def compute_row_statistics(X, prefix=""):
#     Xt = pd.DataFrame()

#     for agg_func in [
#         # "min",
#         # "max",
#         "mean",
#         "std",
#         "kurtosis",
#         "skew",
#     ]:
#         Xt[f"{prefix}{agg_func}"] = X.agg(agg_func, axis=1)

#     return Xt

# + code_folding=[3] execution={"iopub.execute_input": "2020-11-15T04:40:18.021251Z", "iopub.status.busy": "2020-11-15T04:40:18.020509Z", "iopub.status.idle": "2020-11-15T04:40:18.794904Z", "shell.execute_reply": "2020-11-15T04:40:18.794072Z"} papermill={"duration": 0.805422, "end_time": "2020-11-15T04:40:18.795031", "exception": false, "start_time": "2020-11-15T04:40:17.989609", "status": "completed"} tags=[]
from sklearn.metrics import log_loss


def score(Y, Y_pred):
    _, n_classes = Y.shape

    losses = []

    for j in range(n_classes):
        loss = log_loss(Y.iloc[:, j], Y_pred.iloc[:, j], labels=[0, 1])

        losses.append(loss)

    return np.mean(losses)


from sklearn.metrics import roc_auc_score


def auc_score(Y, Y_pred):
    _, n_classes = Y.shape

    aucs = []

    for j in range(n_classes):
        auc = roc_auc_score(Y.iloc[:, j], Y_pred.iloc[:, j])

        aucs.append(auc)

    return np.mean(aucs)


# + code_folding=[7] execution={"iopub.execute_input": "2020-11-15T04:40:18.852597Z", "iopub.status.busy": "2020-11-15T04:40:18.851571Z", "iopub.status.idle": "2020-11-15T04:40:18.855082Z", "shell.execute_reply": "2020-11-15T04:40:18.854311Z"} papermill={"duration": 0.036662, "end_time": "2020-11-15T04:40:18.855208", "exception": false, "start_time": "2020-11-15T04:40:18.818546", "status": "completed"} tags=[]
import os
import random as rn

import tensorflow as tf
import numpy as np


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


# + code_folding=[4] execution={"iopub.execute_input": "2020-11-15T04:40:18.913724Z", "iopub.status.busy": "2020-11-15T04:40:18.912896Z", "iopub.status.idle": "2020-11-15T04:40:18.916149Z", "shell.execute_reply": "2020-11-15T04:40:18.915459Z"} papermill={"duration": 0.037276, "end_time": "2020-11-15T04:40:18.916269", "exception": false, "start_time": "2020-11-15T04:40:18.878993", "status": "completed"} tags=[]
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


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


# + code_folding=[6] execution={"iopub.execute_input": "2020-11-15T04:40:18.982852Z", "iopub.status.busy": "2020-11-15T04:40:18.981785Z", "iopub.status.idle": "2020-11-15T04:40:18.985199Z", "shell.execute_reply": "2020-11-15T04:40:18.984617Z"} _kg_hide-input=false papermill={"duration": 0.045366, "end_time": "2020-11-15T04:40:18.985320", "exception": false, "start_time": "2020-11-15T04:40:18.939954", "status": "completed"} tags=[]
# https://arxiv.org/abs/1905.04899

import numpy as np
import tensorflow as tf


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


# + code_folding=[5] execution={"iopub.execute_input": "2020-11-15T04:40:19.223235Z", "iopub.status.busy": "2020-11-15T04:40:19.222417Z", "iopub.status.idle": "2020-11-15T04:40:19.248983Z", "shell.execute_reply": "2020-11-15T04:40:19.248301Z"} papermill={"duration": 0.069383, "end_time": "2020-11-15T04:40:19.249102", "exception": false, "start_time": "2020-11-15T04:40:19.179719", "status": "completed"} tags=[]
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection._split import _BaseKFold


class MultilabelStratifiedGroupKFold(_BaseKFold):
    def __init__(self, n_splits=5, random_state=None, shuffle=False):
        super().__init__(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

    def _iter_test_indices(self, X=None, Y=None, groups=None):
        cv = MultilabelStratifiedKFold(
            n_splits=self.n_splits,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        value_counts = groups.value_counts()
        regluar_indices = value_counts.loc[
            (value_counts == 6) | (value_counts == 12) | (value_counts == 18)
        ].index.sort_values()
        irregluar_indices = value_counts.loc[
            (value_counts != 6) & (value_counts != 12) & (value_counts != 18)
        ].index.sort_values()

        group_to_fold = {}
        tmp = Y.groupby(groups).mean().loc[regluar_indices]

        for fold, (_, test) in enumerate(cv.split(tmp, tmp)):
            group_to_fold.update({group: fold for group in tmp.index[test]})

        sample_to_fold = {}
        tmp = Y.loc[groups.isin(irregluar_indices)]

        for fold, (_, test) in enumerate(cv.split(tmp, tmp)):
            sample_to_fold.update({sample: fold for sample in tmp.index[test]})

        folds = groups.map(group_to_fold)
        is_na = folds.isna()
        folds[is_na] = folds[is_na].index.map(sample_to_fold).values

        for i in range(self.n_splits):
            yield np.where(folds == i)[0]


# + execution={"iopub.execute_input": "2020-11-15T04:40:19.311784Z", "iopub.status.busy": "2020-11-15T04:40:19.310901Z", "iopub.status.idle": "2020-11-15T04:40:24.963053Z", "shell.execute_reply": "2020-11-15T04:40:24.964132Z"} papermill={"duration": 5.690611, "end_time": "2020-11-15T04:40:24.964366", "exception": false, "start_time": "2020-11-15T04:40:19.273755", "status": "completed"} tags=[]
# dtype = {"cp_type": "category", "cp_dose": "category"}
# index_col = "sig_id"
#
# train_features = pd.read_csv(
#    "../input/lish-moa/train_features.csv", dtype=dtype, index_col=index_col
# )
# X = train_features.select_dtypes("number")
## Y_nonscored = pd.read_csv(
##     "../input/lish-moa/train_targets_nonscored.csv", index_col=index_col
## )
# Y = pd.read_csv("../input/lish-moa/train_targets_scored.csv", index_col=index_col)
# groups = pd.read_csv(
#    "../input/lish-moa/train_drug.csv", index_col=index_col, squeeze=True
# )
#
# columns = Y.columns

# +
dtype = {"cp_type": "category", "cp_dose": "category"}
index_col = "sig_id"

sys.path.append(r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\code")
import datasets

DATADIR = datasets.DATADIR

groups = pd.read_csv(
    f"{DATADIR}/train_drug.csv", dtype=dtype, index_col=index_col, squeeze=True
)
train_features = pd.read_csv(
    f"{DATADIR}/train_features.csv", dtype=dtype, index_col=index_col
)
# X_test = pd.read_csv(f"{DATADIR}/test_features.csv", dtype=dtype, index_col=index_col)
X = train_features.select_dtypes("number")
Y_nonscored = pd.read_csv(f"{DATADIR}/train_targets_nonscored.csv", index_col=index_col)
Y = pd.read_csv(f"{DATADIR}/train_targets_scored.csv", index_col=index_col)

columns = Y.columns

# + execution={"iopub.execute_input": "2020-11-15T04:40:25.034407Z", "iopub.status.busy": "2020-11-15T04:40:25.033563Z", "iopub.status.idle": "2020-11-15T04:40:27.251916Z", "shell.execute_reply": "2020-11-15T04:40:27.251250Z"} papermill={"duration": 2.253691, "end_time": "2020-11-15T04:40:27.252051", "exception": false, "start_time": "2020-11-15T04:40:24.998360", "status": "completed"} tags=[]
clipped_features = ClippedFeatures()
X = clipped_features.fit_transform(X)

with open("clipped_features.pkl", "wb") as f:
    pickle.dump(clipped_features, f)

# c_prefix = "c-"
# g_prefix = "g-"
# c_columns = X.columns.str.startswith(c_prefix)
# g_columns = X.columns.str.startswith(g_prefix)
# X_c = compute_row_statistics(X.loc[:, c_columns], prefix=c_prefix)
# X_g = compute_row_statistics(X.loc[:, g_columns], prefix=g_prefix)
# X = pd.concat([X, X_c, X_g], axis=1)

# Y_nonscored = Y_nonscored.loc[:, Y_nonscored.sum(axis=0) > 0]
# Y = pd.concat([Y, Y_nonscored], axis=1)

# + execution={"iopub.execute_input": "2020-11-15T04:40:27.307722Z", "iopub.status.busy": "2020-11-15T04:40:27.306775Z", "iopub.status.idle": "2020-11-15T04:40:27.309988Z", "shell.execute_reply": "2020-11-15T04:40:27.309231Z"} papermill={"duration": 0.033192, "end_time": "2020-11-15T04:40:27.310108", "exception": false, "start_time": "2020-11-15T04:40:27.276916", "status": "completed"} tags=[]
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape
print(f"n_classes: {n_classes}")


# + code_folding=[] execution={"iopub.execute_input": "2020-11-15T04:40:27.492218Z", "iopub.status.busy": "2020-11-15T04:40:27.491480Z", "iopub.status.idle": "2020-11-15T10:01:54.657676Z", "shell.execute_reply": "2020-11-15T10:01:54.658434Z"} papermill={"duration": 19287.2044, "end_time": "2020-11-15T10:01:54.658768", "exception": false, "start_time": "2020-11-15T04:40:27.454368", "status": "completed"} tags=[]
def train_and_evaluate(model_type="rs", params={}):
    counts = np.empty((n_seeds * n_splits, n_classes))

    bias_initializer = -Y.mean(axis=0).apply(np.log).values
    bias_initializer = tf.keras.initializers.Constant(bias_initializer)

    Y_pred = np.zeros((train_size, n_classes))
    Y_pred = pd.DataFrame(Y_pred, columns=Y.columns, index=Y.index)

    for i in range(n_seeds):
        set_seed(seed=i)

        cv = MultilabelStratifiedGroupKFold(
            n_splits=n_splits, random_state=i, shuffle=shuffle
        )

        # for j, (train, valid) in enumerate(cv.split(X, Y, groups)):
        for j, (train, valid) in enumerate(cv.split(X, Y[columns], groups)):
            counts[i * n_splits + j] = Y.iloc[train].sum()

            os.makedirs(model_type, exist_ok=True)

            # model_nonscored_path = f"model_nonscored_seed_{i}_fold_{j}.h5"
            model_path = f"{model_type}/model_seed_{i}_fold_{j}.h5"

            K.clear_session()
            if model_type == "5l":
                # model = create_model_5l(n_features, num_classes=n_classes, **params)
                model = create_model_5l_v2(n_features, num_classes=n_classes, **params)
            elif model_type == "4l":
                # model = create_model_4l(n_features, num_classes=n_classes)
                model = create_model_4l_v2(n_features, num_classes=n_classes, **params)
            elif model_type == "3l_v2":
                # model = create_model_3l_v2(n_features, num_classes=n_classes)
                model = create_model_3l_v3(n_features, num_classes=n_classes, **params)
            elif model_type == "2l":
                # model = create_model_2l(n_features, num_classes=n_classes)
                model = create_model_2l_v2(n_features, num_classes=n_classes, **params)
            elif model_type == "rs":
                # model = create_model_rs(n_features, len(start_predictors), num_classes=n_classes)
                model = create_model_rs_v2(
                    n_features, len(start_predictors), num_classes=n_classes, **params
                )
            elif model_type == "stacked_tabnet":
                # model = create_model_stacked_tabnet(n_features, num_classes=n_classes)
                model = create_model_stacked_tabnet_v2(
                    n_features, params, num_classes=n_classes
                )

            if model_type == "rs":
                # 入力2つのNN使うから工夫してる
                X_ = X[start_predictors]

                callbacks = build_callbacks(
                    model_path, factor=factor, patience=patience
                )
                history = model.fit(
                    [X.iloc[train], X_.iloc[train]],
                    Y.iloc[train],
                    batch_size=batch_size,
                    callbacks=callbacks,
                    validation_data=([X.iloc[valid], X_.iloc[valid]], Y.iloc[valid]),
                    **fit_params,
                )

                model.load_weights(model_path)

                Y_pred.iloc[valid] += (
                    model.predict([X.iloc[valid], X_.iloc[valid]]) / n_seeds
                )

            else:
                generator = Cutmix(
                    X.iloc[train], Y.iloc[train], alpha=alpha, batch_size=batch_size
                )
                callbacks = build_callbacks(
                    model_path, factor=factor, patience=patience
                )
                history = model.fit(
                    generator,
                    callbacks=callbacks,
                    validation_data=(X.iloc[valid], Y.iloc[valid]),
                    **fit_params,
                )

                model.load_weights(model_path)

                Y_pred.iloc[valid] += model.predict(X.iloc[valid]) / n_seeds

    Y_pred[train_features["cp_type"] == "ctl_vehicle"] = 0.0

    with open("counts.pkl", "wb") as f:
        pickle.dump(counts, f)

    with open(f"Y_pred_{model_type}.pkl", "wb") as f:
        pickle.dump(Y_pred[columns], f)

    oof_score = score(Y[columns], Y_pred[columns])
    print(f"\noof_score: {oof_score}")

    oof_auc_score = auc_score(Y[columns], Y_pred[columns])
    print(f"oof_auc_score: {oof_auc_score}")

    print("-" * 100)

    return oof_score, Y_pred


# + execution={"iopub.execute_input": "2020-11-15T04:40:27.368519Z", "iopub.status.busy": "2020-11-15T04:40:27.367596Z", "iopub.status.idle": "2020-11-15T04:40:27.370793Z", "shell.execute_reply": "2020-11-15T04:40:27.370085Z"} papermill={"duration": 0.035715, "end_time": "2020-11-15T04:40:27.370915", "exception": false, "start_time": "2020-11-15T04:40:27.335200", "status": "completed"} tags=[]
# hyperparameters
alpha = 4.0
batch_size = 1024  # 32
factor = 0.5
n_seeds = 1  # 5
n_splits = 5
patience = 30
shuffle = True
params = {
    "activation": "elu",
    "kernel_initializer": "he_normal",
    "label_smoothing": 5e-04,
    "lr": 0.03,
    "n_layers": 6,
    "n_units": 256,
    "rate": 0.3,
}

stacked_tabnet_params = {
    "batch_momentum": 0.95,
    "feature_dim": 512,
    "norm_type": "batch",
    "num_decision_steps": 1,
    "num_layers": 2,
}

# fit_params = {"epochs": 1_000, "verbose": 0}
fit_params = {"epochs": 50, "verbose": 2}

# DEBUG = True
DEBUG = False
if DEBUG:
    batch_size = 1024
    n_seeds = 2
    n_splits = 2
    fit_params = {"epochs": 2, "verbose": 1}
    print("DEBUG")

# + code_folding=[0]
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
