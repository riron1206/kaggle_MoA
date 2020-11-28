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

# !pwd
import sys

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
sys.executable

# + execution={"iopub.execute_input": "2020-11-03T17:04:57.398778Z", "iopub.status.busy": "2020-11-03T17:04:57.397969Z", "iopub.status.idle": "2020-11-03T17:05:11.690550Z", "shell.execute_reply": "2020-11-03T17:05:11.691118Z"} papermill={"duration": 14.320085, "end_time": "2020-11-03T17:05:11.691314", "exception": false, "start_time": "2020-11-03T17:04:57.371229", "status": "completed"} tags=[]
# #!pip install adabelief-tf==0.1.0

# + execution={"iopub.execute_input": "2020-11-03T17:05:11.786806Z", "iopub.status.busy": "2020-11-03T17:05:11.785909Z", "iopub.status.idle": "2020-11-03T17:05:11.789005Z", "shell.execute_reply": "2020-11-03T17:05:11.788431Z"} papermill={"duration": 0.052264, "end_time": "2020-11-03T17:05:11.789122", "exception": false, "start_time": "2020-11-03T17:05:11.736858", "status": "completed"} tags=[]
import sys

# sys.path.append("../input/iterative-stratification/iterative-stratification-master")

# +
## GPU使わない
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" execution={"iopub.execute_input": "2020-11-03T17:05:11.884813Z", "iopub.status.busy": "2020-11-03T17:05:11.883792Z", "iopub.status.idle": "2020-11-03T17:05:18.486925Z", "shell.execute_reply": "2020-11-03T17:05:18.486133Z"} papermill={"duration": 6.652616, "end_time": "2020-11-03T17:05:18.487048", "exception": false, "start_time": "2020-11-03T17:05:11.834432", "status": "completed"} tags=[]
import os
import glob
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
# -

from funcs import *

logger = Logger("./")

# + code_folding=[20]
import tensorflow as tf
from adabelief_tf import AdaBeliefOptimizer


def get_optimizers(
    choice_optim="sgd",
    lr=0.01,
    decay=0.0,
    momentum=0.9,
    nesterov=True,  # SGD
    rmsprop_rho=0.9,  # RMSprop
    adadelta_rho=0.95,  # Adadelta
    beta_1=0.9,
    beta_2=0.999,
    amsgrad=False,  # Adam, Adamax, Nadam
    total_steps=0,
    warmup_proportion=0.1,
    min_lr=0.0,  # RAdam
    *args,
    **kwargs,
):
    """
    オプティマイザを取得する
    引数のchoice_optim は 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam', 'adabelief', 'radam' のいずれかを指定する
    """
    optim = ""
    if choice_optim == "sgd":
        optim = tf.optimizers.SGD(
            lr=lr, momentum=momentum, decay=decay, nesterov=nesterov
        )
    elif choice_optim == "rmsprop":
        optim = tf.optimizers.RMSprop(lr=lr, rho=rmsprop_rho, decay=decay)
    elif choice_optim == "adagrad":
        optim = tf.optimizers.Adagrad(lr=lr, decay=decay)
    elif choice_optim == "adadelta":
        optim = tf.optimizers.Adadelta(lr=lr, decay=decay, rho=adadelta_rho)
    elif choice_optim == "adam":
        optim = tf.optimizers.Adam(
            lr=lr, decay=decay, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad
        )
    elif choice_optim == "adamax":
        optim = tf.optimizers.Adamax(lr=lr, decay=decay, beta_1=beta_1, beta_2=beta_2)
    elif choice_optim == "nadam":
        if decay == 0.0:
            # Nadam はschedule_decay=0.004 がデフォルト値
            schedule_decay = 0.004
        else:
            schedule_decay = decay
        optim = tf.optimizers.Nadam(
            lr=lr, schedule_decay=schedule_decay, beta_1=beta_1, beta_2=beta_2
        )
    elif choice_optim == "radam":
        if lr == 0.0:
            lr = 0.001
        optim = tfa.optimizers.RectifiedAdam(
            learning_rate=lr,
            weight_decay=decay,
            beta_1=beta_1,
            beta_2=beta_2,
            amsgrad=amsgrad,
            total_steps=total_steps,
            warmup_proportion=warmup_proportion,
            min_lr=min_lr,
        )
    elif choice_optim == "adabelief":
        optim = AdaBeliefOptimizer(learning_rate=lr,)
    return optim


# + code_folding=[22]
def build_model(
    input_dim,
    output_dim,
    activation="relu",
    bias_initializer="zeros",
    kernel_initializer="glorot_uniform",
    label_smoothing=0.0,
    lr=1e-03,
    n_layers=3,
    n_units="auto",
    pretrained_model_path=None,
    rate=0.0,
    skip=False,
    stddev=0.0,
    other_params={
        "optimizer": "adabelief",
        "is_lookahead": False,
        "is_swa": False,
        "is_weightnorm": True,
        "nn_order": "batch-act-drop",
        "drop_act_rate": 0.0,
    },
):
    if n_units == "auto":
        n_units = 0.5 * (input_dim + output_dim)

    inputs = tf.keras.layers.Input(shape=input_dim, name="I")

    x = inputs
    if stddev > 0.0:
        x = tf.keras.layers.GaussianNoise(stddev, name="GN")(x, training=True)

    x = tf.keras.layers.Dropout(rate, name="D0")(x)

    for i in range(n_layers - 2):
        if other_params["is_weightnorm"]:
            x = tfa.layers.WeightNormalization(
                tf.keras.layers.Dense(n_units, kernel_initializer=kernel_initializer),
                name=f"WN{i + 1}",
            )(x)
        else:
            x = tf.keras.layers.Dense(
                n_units, kernel_initializer=kernel_initializer, name=f"DN{i + 1}"
            )(x)

        # batch-act-drop
        if other_params["nn_order"] == "batch-act-drop":
            x = tf.keras.layers.BatchNormalization(name=f"BN{i + 1}")(x)

            if skip and i > 0 and i % 2 == 0:
                x = x + shortcut

            if other_params["drop_act_rate"] > 0.0:
                x = DropActivation(
                    activation, name=f"A{i + 1}", rate=other_params["drop_act_rate"]
                )(x)
            else:
                x = tf.keras.layers.Activation(activation, name=f"A{i + 1}")(x)
                x = tf.keras.layers.Dropout(rate, name=f"D{i + 1}")(x)

            if skip and i % 2 == 0:
                shortcut = x

        # batch-drop-act
        if other_params["nn_order"] == "batch-drop-act":
            x = tf.keras.layers.BatchNormalization(name=f"BN{i + 1}")(x)

            if skip and i > 0 and i % 2 == 0:
                x = x + shortcut

            if other_params["drop_act_rate"] > 0.0:
                x = DropActivation(
                    activation, name=f"A{i + 1}", rate=other_params["drop_act_rate"]
                )(x)
            else:
                x = tf.keras.layers.Dropout(rate, name=f"D{i + 1}")(x)
                x = tf.keras.layers.Activation(activation, name=f"A{i + 1}")(x)

            if skip and i % 2 == 0:
                shortcut = x

        # drop-act-batch
        if other_params["nn_order"] == "drop-act-batch":
            if other_params["drop_act_rate"] <= 0.0:
                x = tf.keras.layers.Dropout(rate, name=f"D{i + 1}")(x)

            if skip and i > 0 and i % 2 == 0:
                x = x + shortcut

            if other_params["drop_act_rate"] > 0.0:
                x = DropActivation(
                    activation, name=f"A{i + 1}", rate=other_params["drop_act_rate"]
                )(x)
            else:
                x = tf.keras.layers.Activation(activation, name=f"A{i + 1}")(x)
            x = tf.keras.layers.BatchNormalization(name=f"BN{i + 1}")(x)

            if skip and i % 2 == 0:
                shortcut = x

        # drop-batch-act
        if other_params["nn_order"] == "drop-batch-act":
            if other_params["drop_act_rate"] <= 0.0:
                x = tf.keras.layers.Dropout(rate, name=f"D{i + 1}")(x)

            if skip and i > 0 and i % 2 == 0:
                x = x + shortcut

            x = tf.keras.layers.BatchNormalization(name=f"BN{i + 1}")(x)
            if other_params["drop_act_rate"] > 0.0:
                x = DropActivation(
                    activation, name=f"A{i + 1}", rate=other_params["drop_act_rate"]
                )(x)
            else:
                x = tf.keras.layers.Activation(activation, name=f"A{i + 1}")(x)

            if skip and i % 2 == 0:
                shortcut = x

        # act-batch-drop
        if other_params["nn_order"] == "act-batch-drop":
            if other_params["drop_act_rate"] > 0.0:
                x = DropActivation(
                    activation, name=f"A{i + 1}", rate=other_params["drop_act_rate"]
                )(x)
            else:
                x = tf.keras.layers.Activation(activation, name=f"A{i + 1}")(x)

            if skip and i > 0 and i % 2 == 0:
                x = x + shortcut

            x = tf.keras.layers.BatchNormalization(name=f"BN{i + 1}")(x)
            if other_params["drop_act_rate"] <= 0.0:
                x = tf.keras.layers.Dropout(rate, name=f"D{i + 1}")(x)

            if skip and i % 2 == 0:
                shortcut = x

        # act-drop-batch
        if other_params["nn_order"] == "act-drop-batch":
            if other_params["drop_act_rate"] > 0.0:
                x = DropActivation(
                    activation, name=f"A{i + 1}", rate=other_params["drop_act_rate"]
                )(x)
            else:
                x = tf.keras.layers.Activation(activation, name=f"A{i + 1}")(x)

            if skip and i > 0 and i % 2 == 0:
                x = x + shortcut

            if other_params["drop_act_rate"] <= 0.0:
                x = tf.keras.layers.Dropout(rate, name=f"D{i + 1}")(x)
            x = tf.keras.layers.BatchNormalization(name=f"BN{i + 1}")(x)

            if skip and i % 2 == 0:
                shortcut = x

    if other_params["is_weightnorm"]:
        x = tfa.layers.WeightNormalization(
            tf.keras.layers.Dense(output_dim, bias_initializer=bias_initializer),
        )(x)
    else:
        x = tf.keras.layers.Dense(output_dim, bias_initializer=bias_initializer)(x)

    outputs = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    if pretrained_model_path is not None:
        model.load_weights(pretrained_model_path, by_name=True, skip_mismatch=True)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)

    optimizer = get_optimizers(choice_optim=other_params["optimizer"], lr=lr)
    if other_params["is_lookahead"]:
        optimizer = tfa.optimizers.Lookahead(optimizer)
    if other_params["is_swa"]:
        optimizer = tfa.optimizers.SWA(optimizer)

    model.compile(loss=loss, optimizer=optimizer)

    return model


# + code_folding=[5]
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection._split import _BaseKFold


class MultilabelGroupStratifiedKFold(_BaseKFold):
    def __init__(self, n_splits=5, random_state=None, shuffle=False):
        super().__init__(n_splits=n_splits, random_state=random_state, shuffle=shuffle)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        cv = MultilabelStratifiedKFold(
            n_splits=self.n_splits,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        value_counts = groups.value_counts()
        regular_index = value_counts.loc[
            (value_counts == 6) | (value_counts == 12) | (value_counts == 18)
        ].index.sort_values()
        irregular_index = value_counts.loc[
            (value_counts != 6) & (value_counts != 12) & (value_counts != 18)
        ].index.sort_values()

        group_to_fold = {}
        tmp = Y.groupby(groups).mean().loc[regular_index]

        for fold, (_, test) in enumerate(cv.split(tmp, tmp)):
            group_to_fold.update({group: fold for group in tmp.index[test]})

        sample_to_fold = {}
        tmp = Y.loc[groups.isin(irregular_index)]

        for fold, (_, test) in enumerate(cv.split(tmp, tmp)):
            sample_to_fold.update({sample: fold for sample in tmp.index[test]})

        folds = groups.map(group_to_fold)
        is_na = folds.isna()
        folds[is_na] = folds[is_na].index.map(sample_to_fold).values

        for i in range(self.n_splits):
            yield np.where(folds == i)[0]


# + code_folding=[21]
def preprocess(
    X,
    Y,
    Y_nonscored,
    X_test,
    preprocess_params={
        "is_clip": True,
        "is_stat": True,
        "is_add_nonscored": True,
        "is_c_squared": False,
        "is_c_abs": False,
        "is_g_valid": False,
        "pca_n_components_g": 0,
        "pca_n_components_c": 0,
        "is_scaling": False,
        "quantile_n_components": 0,
        "fe_cluster_n_clusters_g": 0,
        "fe_cluster_n_clusters_c": 0,
        "is_fit_train_only": True,
        "variance_threshold": 0.0,
    },
):
    if preprocess_params["is_clip"]:
        clipped_features = ClippedFeatures()
        X = clipped_features.fit_transform(X)

        # with open("clipped_features.pkl", "wb") as f:
        #    pickle.dump(clipped_features, f)

    if preprocess_params["quantile_n_components"] > 0:
        X, X_test = fe_quantile_transformer(
            X,
            X_test,
            n_quantiles=preprocess_params["quantile_n_components"],
            random_state=0,
        )

    if preprocess_params["fe_cluster_n_clusters_g"] > 0:
        X, X_test, _ = fe_cluster(
            X,
            X_test,
            n_clusters_g=preprocess_params["fe_cluster_n_clusters_g"],
            n_clusters_c=preprocess_params["fe_cluster_n_clusters_c"],
            random_state=123,
            is_fit_train_only=preprocess_params["is_fit_train_only"],
        )

    if preprocess_params["is_c_squared"]:
        X, X_test, _ = c_squared(X, X_test)

    if preprocess_params["is_c_abs"]:
        X, X_test, _ = c_abs(X, X_test)

    if preprocess_params["is_g_valid"]:
        X, X_test, _ = g_valid(X, X_test)

    if preprocess_params["pca_n_components_g"] > 0:
        X, X_test, _ = fe_pca(
            X,
            X_test,
            n_components_g=preprocess_params["pca_n_components_g"],
            n_components_c=preprocess_params["pca_n_components_c"],
            random_state=123,
            is_fit_train_only=preprocess_params["is_fit_train_only"],
        )

    if preprocess_params["is_scaling"]:
        X, X_test, _ = scaling(
            X, X_test, is_fit_train_only=preprocess_params["is_fit_train_only"],
        )

    if preprocess_params["is_stat"]:
        c_prefix = "c-"
        g_prefix = "g-"
        c_columns = X.columns.str.startswith(c_prefix)
        g_columns = X.columns.str.startswith(g_prefix)
        X_c = compute_row_statistics(X.loc[:, c_columns], prefix=c_prefix)
        X_g = compute_row_statistics(X.loc[:, g_columns], prefix=g_prefix)
        X = pd.concat([X, X_c, X_g], axis=1)

    if preprocess_params["variance_threshold"] > 0.0:
        X, X_test = fe_variance_threshold(
            X,
            X_test,
            get_features(X),
            threshold=preprocess_params["variance_threshold"],
            is_fit_train_only=preprocess_params["is_fit_train_only"],
        )

    if preprocess_params["is_add_nonscored"]:
        Y_nonscored = Y_nonscored.loc[:, Y_nonscored.sum(axis=0) > 0]

        Y = pd.concat([Y, Y_nonscored], axis=1)

    return X, Y, Y_nonscored


# + code_folding=[0]
def pretrain(X, Y_nonscored, is_drug_kfold=True):
    bias_initializer = -Y_nonscored.mean(axis=0).apply(np.log).values
    bias_initializer = tf.keras.initializers.Constant(bias_initializer)

    # Y_pred = np.zeros((train_size, n_classes_nonscored))
    # Y_pred = pd.DataFrame(Y_pred, columns=Y_nonscored.columns, index=X.index)

    for i in range(n_seeds):
        set_seed(seed=i)

        if is_drug_kfold:
            cv = MultilabelGroupStratifiedKFold(
                n_splits=n_splits, random_state=i, shuffle=shuffle
            )
            cv_split = cv.split(X, Y_nonscored, groups)
        else:
            cv = MultilabelStratifiedKFold(
                n_splits=n_splits, random_state=i, shuffle=shuffle
            )
            cv_split = cv.split(X, Y_nonscored)

        for j, (train, valid) in tqdm(enumerate(cv_split)):

            #model_path = f"model_nonscored_seed_{i}_fold_{j}.h5"
            model_path = f"{out_dir}/model_nonscored_seed_{i}_fold_{j}.h5"

            K.clear_session()
            model = build_model(
                n_features,
                n_classes_nonscored,
                bias_initializer=bias_initializer,
                **params,
            )

            callbacks = build_callbacks(model_path, factor=factor, patience=patience)
            history = model.fit(
                X.iloc[train],
                Y_nonscored.iloc[train],
                callbacks=callbacks,
                validation_data=(X.iloc[valid], Y_nonscored.iloc[valid]),
                **fit_params,
            )

            # model.load_weights(model_path)

            # Y_pred.iloc[valid] += predict(model, X.iloc[valid], **predict_params) / n_seeds

    # X = pd.concat([X, Y_pred], axis=1)
    # n_features += n_classes_nonscored


# + code_folding=[]
def train_and_evaluate(X, Y, is_ctl_vehicle=False, is_drug_kfold=True, is_pretrain=False):
    bias_initializer = -Y.mean(axis=0).apply(np.log).values
    bias_initializer = tf.keras.initializers.Constant(bias_initializer)

    Y_pred = np.zeros((train_size, n_classes))
    Y_pred = pd.DataFrame(Y_pred, columns=Y.columns, index=X.index)

    for i in range(n_seeds):
        set_seed(seed=i)

        if is_drug_kfold:
            cv = MultilabelGroupStratifiedKFold(
                n_splits=n_splits, random_state=i, shuffle=shuffle
            )
            cv_split = cv.split(X, Y, groups)
        else:
            cv = MultilabelStratifiedKFold(
                n_splits=n_splits, random_state=i, shuffle=shuffle
            )
            cv_split = cv.split(X, Y)

        for j, (train, valid) in tqdm(enumerate(cv_split)):

            if is_pretrain:
                # model_nonscored_path = f"model_nonscored_seed_{i}_fold_{j}.h5"
                model_path = f"{out_dir}/model_nonscored_seed_{i}_fold_{j}.h5"
                params["pretrained_model_path"] = model_path
            
            # model_path = f"model_seed_{i}_fold_{j}.h5"
            model_path = f"{out_dir}/model_seed_{i}_fold_{j}.h5"

            K.clear_session()
            model = build_model(
                n_features,
                n_classes,
                bias_initializer=bias_initializer,
                # pretrained_model_path=model_nonscored_path,
                **params,
            )

            callbacks = build_callbacks(model_path, factor=factor, patience=patience)

            if cutmix_alpha > 0.0:
                history = model.fit(
                    Cutmix(
                        X.iloc[train],
                        y=Y.iloc[train],
                        batch_size=fit_params["batch_size"],
                        alpha=cutmix_alpha,
                    ),
                    callbacks=callbacks,
                    validation_data=(X.iloc[valid], Y.iloc[valid]),
                    **fit_params,
                )
            elif mixup_alpha > 0.0:
                history = model.fit(
                    Mixup(
                        X.iloc[train],
                        y=Y.iloc[train],
                        batch_size=fit_params["batch_size"],
                        alpha=mixup_alpha,
                    ),
                    callbacks=callbacks,
                    validation_data=(X.iloc[valid], Y.iloc[valid]),
                    **fit_params,
                )
            else:
                history = model.fit(
                    X.iloc[train],
                    Y.iloc[train],
                    callbacks=callbacks,
                    validation_data=(X.iloc[valid], Y.iloc[valid]),
                    **fit_params,
                )

            plot_history(history)

            model.load_weights(model_path)

            Y_pred.iloc[valid] += (
                predict(model, X.iloc[valid], **predict_params) / n_seeds
            )

    # postprocess
    if is_ctl_vehicle:
        Y_pred = Y_pred.append(
            Y_orig[train_features["cp_type"] == "ctl_vehicle"]
        ).sort_index()
        Y = Y_orig.sort_index()
    else:
        Y_pred[train_features["cp_type"] == "ctl_vehicle"] = 0.0

    Y_pred[Y_pred < threshold] = 0.0
    Y_pred[Y_pred > 1.0 - threshold] = 1.0

    oof_loss = score(Y[columns], Y_pred[columns])
    print(f"Our out of folds log loss score is {oof_loss}")

    return oof_loss


# + code_folding=[]
DEBUG = True
#DEBUG = False

out_base_dir = "output"
os.makedirs(out_base_dir, exist_ok=True)

# hyperparameters
batch_size = 32
factor = 0.5
n_seeds = 1  # n_seeds = 5
n_splits = 5
patience = 30
shuffle = True
threshold = 1e-05
params = {
    "activation": "elu",
    "kernel_initializer": "he_normal",
    "label_smoothing": 5e-04,
    "lr": 0.03,
    "n_layers": 7,
    "n_units": 256,
    "rate": 0.25,
    "skip": True,
    "stddev": 0.45,
    "other_params": {
        "optimizer": "adam",
        "is_lookahead": False,
        "is_swa": False,
        "is_weightnorm": True,
        "nn_order": "batch-act-drop",
        "drop_act_rate": 0.0,
    },
}
fit_params = {"batch_size": batch_size, "epochs": 1_000, "verbose": 0}
cutmix_alpha = 0.0
mixup_alpha = 0.0
predict_params = {"batch_size": batch_size, "n_iter": 10}

if DEBUG:
    fit_params = {"batch_size": 1024, "epochs": 80, "verbose": 0}
    logger.info(f"### fit_params: {fit_params} ###")
else:
    logger.info(f"### fit_params: {fit_params} ###")


# preprocess
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(X, Y, Y_nonscored, X_test)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape
# -

# # activation, bn, dropout の順序

# + code_folding=[]
# train
for nn_order in [
    "batch-act-drop",
    "batch-drop-act",
    "drop-batch-act",
    "drop-act-batch",
    "act-batch-drop",
    "act-drop-batch",
]:

    out_dir = f"{out_base_dir}/{nn_order}"
    os.makedirs(out_dir, exist_ok=True)

    params["other_params"]["nn_order"] = nn_order
    oof_loss = train_and_evaluate(X, Y)
    logger.result(f"nn_order: {params['other_params']['nn_order']}\t{oof_loss}")
    print("-" * 100)

params["other_params"]["nn_order"] = "batch-act-drop"
# -
# # weight normalization の有無


# +
# train
for is_weightnorm in [True, False]:

    out_dir = f"{out_base_dir}/is_weightnorm_{is_weightnorm}"
    os.makedirs(out_dir, exist_ok=True)

    params["other_params"]["is_weightnorm"] = is_weightnorm
    oof_loss = train_and_evaluate(X, Y)
    logger.result(
        f"is_weightnorm: {params['other_params']['is_weightnorm']}\t{oof_loss}"
    )
    print("-" * 100)

params["other_params"]["is_weightnorm"] = True
# -

# # DropActivation の有無

# +
# train
for drop_act_rate in [0.05, 0.25]:

    out_dir = f"{out_base_dir}/DropActivation"
    os.makedirs(out_dir, exist_ok=True)

    params["other_params"]["drop_act_rate"] = drop_act_rate
    oof_loss = train_and_evaluate(X, Y)
    logger.result(
        f"drop_act_rate: {params['other_params']['drop_act_rate']}\t{oof_loss}"
    )
    print("-" * 100)

params["other_params"]["drop_act_rate"] = 0.0
# -

# # CutMix, MixUp

# +
# train
for cutmix_alpha in [0.3, 1.0]:

    out_dir = f"{out_base_dir}"
    os.makedirs(out_dir, exist_ok=True)

    oof_loss = train_and_evaluate(X, Y)
    logger.result(f"cutmix_alpha: {cutmix_alpha}\t{oof_loss}")
    print("-" * 100)

cutmix_alpha = 0.0

# +
# train
params["stddev"] = 0.0
cutmix_alpha = 1.0

out_dir = f"{out_base_dir}"
os.makedirs(out_dir, exist_ok=True)

oof_loss = train_and_evaluate(X, Y)
logger.result(f"stddev: {params['stddev']}, cutmix_alpha: {cutmix_alpha}\t{oof_loss}")
print("-" * 100)

params["stddev"] = 0.45
cutmix_alpha = 0.0

# +
# train
for mixup_alpha in [0.2, 0.5]:

    out_dir = f"{out_base_dir}"
    os.makedirs(out_dir, exist_ok=True)

    oof_loss = train_and_evaluate(X, Y)
    logger.result(f"mixup_alpha: {mixup_alpha}\t{oof_loss}")
    print("-" * 100)

mixup_alpha = 0.0

# +
# train
params["stddev"] = 0.0
mixup_alpha = 0.5

out_dir = f"{out_base_dir}"
os.makedirs(out_dir, exist_ok=True)

oof_loss = train_and_evaluate(X, Y)
logger.result(f"stddev: {params['stddev']}, mixup_alpha: {mixup_alpha}\t{oof_loss}")
print("-" * 100)

params["stddev"] = 0.45
mixup_alpha = 0.0
# -

# # pretrain
# - https://www.kaggle.com/c/lish-moa/discussion/195859 より

# +
# train
params["stddev"] = 0.0
cutmix_alpha = 1.0

out_dir = f"{out_base_dir}"

pretrain(X, Y_nonscored)
oof_loss = train_and_evaluate(X, Y, is_pretrain=True)
logger.result(f"pretrain, stddev: {params['stddev']}, cutmix_alpha: {cutmix_alpha}\t{oof_loss}")
print("-" * 100)

params["stddev"] = 0.45
cutmix_alpha = 0.0
# -

# # cp_type と cp_dose の有無

# +
dtype = {}
index_col = "sig_id"

DATADIR = (
    r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\input\lish-moa"
)
# DATADIR = r"../input/lish-moa"

_train_features = pd.read_csv(
    f"{DATADIR}/train_features.csv", dtype=dtype, index_col=index_col
)

X = _train_features.copy()
cp_type = {"trt_cp": 0, "ctl_vehicle": 1}
cp_dose = {"D1": 0, "D2": 1}
for _X in [X, X_test]:
    _X["cp_type"] = _X["cp_type"].map(cp_type)
    _X["cp_dose"] = _X["cp_dose"].map(cp_dose)
X = X.select_dtypes("number")

X, Y, Y_nonscored = preprocess(X, Y, Y_nonscored, X_test)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

display(X.head())

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"cp_type + cp_dose\t{oof_loss}")
print("-" * 100)

# +
dtype = {}
index_col = "sig_id"
DATADIR = (
    r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\input\lish-moa"
)
# DATADIR = r"../input/lish-moa"
_train_features = pd.read_csv(
    f"{DATADIR}/train_features.csv", dtype=dtype, index_col=index_col
)
X = _train_features.copy()
cp_type = {"trt_cp": 0, "ctl_vehicle": 1}
for _X in [X, X_test]:
    _X["cp_type"] = _X["cp_type"].map(cp_type)
X = X.select_dtypes("number")

X, Y, Y_nonscored = preprocess(X, Y, Y_nonscored, X_test)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

display(X.head())

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"cp_type\t{oof_loss}")
print("-" * 100)

# +
dtype = {}
index_col = "sig_id"
DATADIR = (
    r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\input\lish-moa"
)
# DATADIR = r"../input/lish-moa"
_train_features = pd.read_csv(
    f"{DATADIR}/train_features.csv", dtype=dtype, index_col=index_col
)
X = _train_features.copy()
cp_dose = {"D1": 0, "D2": 1}
for _X in [X, X_test]:
    _X["cp_dose"] = _X["cp_dose"].map(cp_dose)
X = X.select_dtypes("number")

X, Y, Y_nonscored = preprocess(X, Y, Y_nonscored, X_test)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

display(X.head())

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"cp_dose\t{oof_loss}")
print("-" * 100)
# -

# # 特徴量指定

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
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X = X.loc[:,start_predictors]

X, Y, Y_nonscored = preprocess(X, Y, Y_nonscored, X_test)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

display(X.head())

# train
params["stddev"] = 0.0
cutmix_alpha = 1.0

out_dir = f"{out_base_dir}"

oof_loss = train_and_evaluate(X, Y)
logger.result(f"start_predictors, stddev: {params['stddev']}, cutmix_alpha: {cutmix_alpha}\t{oof_loss}")
print("-" * 100)

params["stddev"] = 0.45
cutmix_alpha = 0.0
# -

# # 前処理・特徴エンジニアリング

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 600,
    "pca_n_components_c": 50,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 600,
    "pca_n_components_c": 50,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": False,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 70,
    "pca_n_components_c": 10,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 70,
    "pca_n_components_c": 10,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": False,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 100,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 100,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": False,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

# train pretrainあり
params["stddev"] = 0.0
cutmix_alpha = 1.0

out_dir = f"{out_base_dir}"

pretrain(X, Y_nonscored)
oof_loss = train_and_evaluate(X, Y)
logger.result(f"pretrain, stddev: {params['stddev']}, cutmix_alpha: {cutmix_alpha}, preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

params["stddev"] = 0.45
cutmix_alpha = 0.0

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": True,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": True,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": True,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": True,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 70,
    "pca_n_components_c": 10,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 600,
    "pca_n_components_c": 50,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": True,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 100,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 100,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": False,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 1000,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": False,
    "is_stat": False,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 1000,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": False,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": True,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 600,
    "pca_n_components_c": 50,
    "is_scaling": False,
    "quantile_n_components": 100,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": True,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 600,
    "pca_n_components_c": 50,
    "is_scaling": False,
    "quantile_n_components": 100,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.8,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": True,
    "is_c_squared": True,
    "is_c_abs": True,
    "is_g_valid": True,
    "pca_n_components_g": 70,
    "pca_n_components_c": 10,
    "is_scaling": True,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": True,
    "is_c_squared": True,
    "is_c_abs": True,
    "is_g_valid": True,
    "pca_n_components_g": 70,
    "pca_n_components_c": 10,
    "is_scaling": True,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": False,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": True,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": False,
    "is_c_squared": True,
    "is_c_abs": True,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": False,
    "is_c_squared": True,
    "is_c_abs": True,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)
# -


# # ctl_vehicle 行の削除
# - X,Yの行数変わるから最後に実行すること

# +
out_dir = out_base_dir

is_ctl_vehicle = True

X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(X, Y, Y_nonscored, X_test)
Y_orig = Y.copy()
if is_ctl_vehicle:
    X = X[train_features["cp_type"] != "ctl_vehicle"]
    Y = Y[train_features["cp_type"] != "ctl_vehicle"]
    groups = groups[train_features["cp_type"] != "ctl_vehicle"]
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

oof_loss = train_and_evaluate(X, Y, is_ctl_vehicle=is_ctl_vehicle)
logger.result(f"del ctl_vehicle\t{oof_loss}")
print("-" * 100)


# +
out_dir = out_base_dir

is_ctl_vehicle = True

X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(X, Y, Y_nonscored, X_test)
Y_orig = Y.copy()
if is_ctl_vehicle:
    X = X[train_features["cp_type"] != "ctl_vehicle"]
    Y = Y[train_features["cp_type"] != "ctl_vehicle"]
    groups = groups[train_features["cp_type"] != "ctl_vehicle"]
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

params["stddev"] = 0.0
cutmix_alpha = 1.0

oof_loss = train_and_evaluate(X, Y, is_ctl_vehicle=is_ctl_vehicle)
logger.result(f"del ctl_vehicles, tddev: {params['stddev']}, cutmix_alpha: {cutmix_alpha}\t{oof_loss}")
print("-" * 100)
# -

# # 組み合わせ

# +
out_dir = out_base_dir

is_ctl_vehicle = True

X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(X, Y, Y_nonscored, X_test)
Y_orig = Y.copy()
if is_ctl_vehicle:
    X = X[train_features["cp_type"] != "ctl_vehicle"]
    Y = Y[train_features["cp_type"] != "ctl_vehicle"]
    groups = groups[train_features["cp_type"] != "ctl_vehicle"]

params["stddev"] = 0.0
cutmix_alpha = 1.0
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

oof_loss = train_and_evaluate(X, Y, is_ctl_vehicle=is_ctl_vehicle)
logger.result(
    f"del ctl_vehicle, stddev: {params['stddev']}, cutmix_alpha: {cutmix_alpha}, preprocess_params: {preprocess_params}\t{oof_loss}"
)
print("-" * 100)

params["stddev"] = 0.45
cutmix_alpha = 0.0

# +
out_dir = out_base_dir

is_ctl_vehicle = True

X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(X, Y, Y_nonscored, X_test)
Y_orig = Y.copy()
if is_ctl_vehicle:
    X = X[train_features["cp_type"] != "ctl_vehicle"]
    Y = Y[train_features["cp_type"] != "ctl_vehicle"]
    groups = groups[train_features["cp_type"] != "ctl_vehicle"]

params["stddev"] = 0.0
cutmix_alpha = 1.0
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": False,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": False,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

oof_loss = train_and_evaluate(X, Y, is_ctl_vehicle=is_ctl_vehicle)
logger.result(
    f"del ctl_vehicle, stddev: {params['stddev']}, cutmix_alpha: {cutmix_alpha}, preprocess_params: {preprocess_params}\t{oof_loss}"
)
print("-" * 100)

params["stddev"] = 0.45
cutmix_alpha = 0.0
# -



# # no DEBUG

# +
#DEBUG = True
DEBUG = False

out_base_dir = "output"
os.makedirs(out_base_dir, exist_ok=True)

# hyperparameters
batch_size = 32
factor = 0.5
n_seeds = 1  # n_seeds = 5
n_splits = 5
patience = 30
shuffle = True
threshold = 1e-05
params = {
    "activation": "elu",
    "kernel_initializer": "he_normal",
    "label_smoothing": 5e-04,
    "lr": 0.03,
    "n_layers": 7,
    "n_units": 256,
    "rate": 0.25,
    "skip": True,
    "stddev": 0.45,
    "other_params": {
        "optimizer": "adabelief",
        "is_lookahead": False,
        "is_swa": False,
        "is_weightnorm": True,
        "nn_order": "batch-act-drop",
        "drop_act_rate": 0.0,
    },
}
fit_params = {"batch_size": batch_size, "epochs": 1_000, "verbose": 0}
cutmix_alpha = 0.0
mixup_alpha = 0.0
predict_params = {"batch_size": batch_size, "n_iter": 10}

if DEBUG:
    fit_params = {"batch_size": 1024, "epochs": 80, "verbose": 0}
    logger.info(f"### fit_params: {fit_params} ###")
else:
    logger.info(f"### fit_params: {fit_params} ###")


# preprocess
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(X, Y, Y_nonscored, X_test)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

# +
# train
mixup_alpha = 0.5

out_dir = f"{out_base_dir}"
os.makedirs(out_dir, exist_ok=True)

oof_loss = train_and_evaluate(X, Y)
logger.result(f"mixup_alpha: {mixup_alpha}, params: {params}, fit_params: {fit_params}\t{oof_loss}")
print("-" * 100)

mixup_alpha = 0.0

# +
# train
params["stddev"] = 0.0
cutmix_alpha = 1.0

out_dir = f"{out_base_dir}"
os.makedirs(out_dir, exist_ok=True)

oof_loss = train_and_evaluate(X, Y)
logger.result(f"stddev: {params['stddev']}, cutmix_alpha: {cutmix_alpha}, params: {params}, fit_params: {fit_params}\t{oof_loss}")
print("-" * 100)

params["stddev"] = 0.45
cutmix_alpha = 0.0

# +
# preprocess
preprocess_params = {
    "is_clip": True,
    "is_stat": True,
    "is_add_nonscored": False,
    "is_c_squared": False,
    "is_c_abs": True,
    "is_g_valid": False,
    "pca_n_components_g": 0,
    "pca_n_components_c": 0,
    "is_scaling": False,
    "quantile_n_components": 0,
    "fe_cluster_n_clusters_g": 0,
    "fe_cluster_n_clusters_c": 0,
    "is_fit_train_only": True,
    "variance_threshold": 0.0,
}
X, Y, Y_nonscored, train_features, columns, groups, X_test = load_data()
X, Y, Y_nonscored = preprocess(
    X, Y, Y_nonscored, X_test, preprocess_params=preprocess_params
)
train_size, n_features = X.shape
_, n_classes_nonscored = Y_nonscored.shape
_, n_classes = Y.shape

out_dir = out_base_dir

oof_loss = train_and_evaluate(X, Y)
logger.result(f"preprocess_params: {preprocess_params}\t{oof_loss}")
print("-" * 100)
