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

# +
import datetime
import os
import pathlib
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

sys.path.append(r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\code")
import mlp_tf, datasets, util


# + code_folding=[]
def load_data():
    (
        train,
        train_targets,
        test,
        sample_submission,
        train_targets_nonscored,
        train_drug,
    ) = datasets.load_orig_data()
    train, train_targets, test, train_targets_nonscored = datasets.mapping_and_filter(
        train, train_targets, test, train_targets_nonscored
    )
    return (
        train,
        train_targets,
        test,
        sample_submission,
        train.columns[2:],
        train_targets_nonscored,
    )


# -


# ## 条件かえてoof_log_loss確認

(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()

train_targets

train_targets_nonscored

features

logger = util.Logger("./")
# デバッグ用にデータ減らすか
DEBUG = False
#DEBUG = True
if DEBUG:
    mlp_tf.FOLDS = 2  # cvの数
    mlp_tf.EPOCHS = 2
    logger.info(f"### DEBUG: FOLDS:{mlp_tf.FOLDS}, EPOCHS:{mlp_tf.EPOCHS} ###")
else:
    logger.info(f"### HONBAN: FOLDS:{mlp_tf.FOLDS}, EPOCHS:{mlp_tf.EPOCHS} ###")

# ## 動作確認

# +
# 1モデルだけで学習推論できるかテスト
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()

mlp_tf.train_and_evaluate(
    train, test, train_targets, features, train_targets_nonscored, model_type="rs"
)
mlp_tf.inference(
    train, test, train_targets, features, train_targets_nonscored, model_type="rs"
)
# -


str_condition = "no feature_eng"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
# モデルブレンドの学習実行できるかテスト
mlp_tf.run_mlp_tf_blend_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    str_condition=str_condition,
)
# モデルブレンドの推論実行できるかテスト
mlp_tf.run_mlp_tf_blend_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    str_condition=str_condition,
    is_train=False,
)

# ## best FE?

str_condition = (
    "fe_stats flag_add=False → c_squared → c_abs → g_valid → fe_pca → scaling"
)
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.c_abs(train, test)
train, test, features = datasets.g_valid(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
mlp_tf.run_mlp_tf_blend_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    str_condition=str_condition,
)

str_condition = (
    "fe_stats flag_add=False → c_squared → c_abs → g_valid → fe_pca → scaling → fe_clipping"
)
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.c_abs(train, test)
train, test, features = datasets.g_valid(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
features_g, features_c = datasets.get_features_gc(train)
train, test = datasets.fe_clipping(train, test, features_g=features_g, features_c=features_c)
mlp_tf.run_mlp_tf_blend_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_types=[
        "2l",
        "3l",
        "4l",
        "5l",
        "rs",
        "3l_v2",
        "stack_tabnet",
        "tabnet_class",
    ],
    str_condition=str_condition,
)

str_condition = "fe_stats flag_add=False → c_squared → c_abs → g_valid → fe_pca → scaling + seeds=[5, 12, 67]"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.c_abs(train, test)
train, test, features = datasets.g_valid(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
mlp_tf.run_mlp_tf_blend_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    seeds=[5, 12, 67],
    str_condition=str_condition,
    model_types=["rs", "2l", "4l", "5l", "3l", "stack_tabnet"],
)



# ## 1モデルだけで試していく

str_condition = (
    "fe_clipping → fe_stats flag_add=False → c_squared → c_abs → g_valid → fe_pca → scaling"
)
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_clipping(train, test)
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.c_abs(train, test)
train, test, features = datasets.g_valid(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = (
    "fe_stats flag_add=False → c_squared → c_abs → g_valid → fe_pca → scaling → fe_clipping"
)
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.c_abs(train, test)
train, test, features = datasets.g_valid(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
features_g, features_c = datasets.get_features_gc(train)
train, test = datasets.fe_clipping(train, test, features_g=features_g, features_c=features_c)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

# +
str_condition = (
    "no feature_eng + is_del_ctl + is_del_noise_drug"
)
(
    train,
    train_targets,
    test,
    sample_submission,
    train_targets_nonscored,
    train_drug,
) = datasets.load_orig_data()
train, train_targets, test, train_targets_nonscored = datasets.mapping_and_filter(
    train, train_targets, test, train_targets_nonscored, is_del_ctl=True, is_del_noise_drug=True
)
features = datasets.get_features(train)

_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)
# -

str_condition = (
    "is_del_ctl + scaling"
)
(
    train,
    train_targets,
    test,
    sample_submission,
    train_targets_nonscored,
    train_drug,
) = datasets.load_orig_data()
train, train_targets, test, train_targets_nonscored = datasets.mapping_and_filter(
    train, train_targets, test, train_targets_nonscored, is_del_ctl=True
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = (
    "is_del_noise_drug + scaling"
)
(
    train,
    train_targets,
    test,
    sample_submission,
    train_targets_nonscored,
    train_drug,
) = datasets.load_orig_data()
train, train_targets, test, train_targets_nonscored = datasets.mapping_and_filter(
    train, train_targets, test, train_targets_nonscored, is_del_noise_drug=True
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = (
    "is_conat_nonscore + scaling"
)
(
    train,
    train_targets,
    test,
    sample_submission,
    train_targets_nonscored,
    train_drug,
) = datasets.load_orig_data()
train, train_targets, test, train_targets_nonscored = datasets.mapping_and_filter(
    train, train_targets, test, train_targets_nonscored, is_conat_nonscore=True
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

# +
for bs in [8, 16, 32, 64, 128, 256, 1024]:
    mlp_tf.BATCH_SIZE = bs
    str_condition = f"no feature_eng + batch_size={mlp_tf.BATCH_SIZE}"
    (
        train,
        train_targets,
        test,
        sample_submission,
        features,
        train_targets_nonscored,
    ) = load_data()
    _, _ = mlp_tf.run_mlp_tf_logger(
        train,
        test,
        train_targets,
        features,
        train_targets_nonscored,
        model_type="tabnet_class",
        str_condition=str_condition,
    )
    (
        train,
        train_targets,
        test,
        sample_submission,
        features,
        train_targets_nonscored,
    ) = load_data()
    _, _ = mlp_tf.run_mlp_tf_logger(
        train,
        test,
        train_targets,
        features,
        train_targets_nonscored,
        model_type="stack_tabnet",
        str_condition=str_condition,
    )
    
mlp_tf.BATCH_SIZE = 128

# +
for lr in [0.1, 0.01, 0.0001, 0.00001]:
    mlp_tf.LR = lr
    str_condition = f"no feature_eng + learning rate={mlp_tf.LR}"
    (
        train,
        train_targets,
        test,
        sample_submission,
        features,
        train_targets_nonscored,
    ) = load_data()
    _, _ = mlp_tf.run_mlp_tf_logger(
        train,
        test,
        train_targets,
        features,
        train_targets_nonscored,
        model_type="tabnet_class",
        str_condition=str_condition,
    )
    (
        train,
        train_targets,
        test,
        sample_submission,
        features,
        train_targets_nonscored,
    ) = load_data()
    _, _ = mlp_tf.run_mlp_tf_logger(
        train,
        test,
        train_targets,
        features,
        train_targets_nonscored,
        model_type="stack_tabnet",
        str_condition=str_condition,
    )
    
mlp_tf.LR = 0.001
# -

for num_decision_steps in [1, 2, 3, 4, 8]:
    str_condition = f"no feature_eng + num_decision_steps={num_decision_steps}"
    (
        train,
        train_targets,
        test,
        sample_submission,
        features,
        train_targets_nonscored,
    ) = load_data()
    tabnet_params = dict(
            feature_columns=None,
            num_classes=206,
            feature_dim=128,
            output_dim=64,
            num_features=len(features),
            num_decision_steps=num_decision_steps,
            relaxation_factor=1.5,
            sparsity_coefficient=1e-5,
            batch_momentum=0.98,
            virtual_batch_size=None,
            norm_type="group",
            num_groups=-1,
            multi_label=True,
        )
    _, _ = mlp_tf.run_mlp_tf_logger(
        train,
        test,
        train_targets,
        features,
        train_targets_nonscored,
        model_type="tabnet_class",
        str_condition=str_condition,
        tabnet_params=tabnet_params,
    )
    (
        train,
        train_targets,
        test,
        sample_submission,
        features,
        train_targets_nonscored,
    ) = load_data()
    tabnet_params = dict(
            feature_columns=None,
            num_layers=2,
            num_classes=206,
            feature_dim=128,
            output_dim=64,
            num_features=len(features),
            num_decision_steps=num_decision_steps,
            relaxation_factor=1.5,
            sparsity_coefficient=1e-5,
            batch_momentum=0.98,
            virtual_batch_size=None,
            norm_type="group",
            num_groups=-1,
            multi_label=True,
        )
    _, _ = mlp_tf.run_mlp_tf_logger(
        train,
        test,
        train_targets,
        features,
        train_targets_nonscored,
        model_type="stack_tabnet",
        str_condition=str_condition,
        tabnet_params=tabnet_params,
    )

for feature_dim in [8, 32, 64, 128]:
    str_condition = f"no feature_eng + feature_dim,output_dim={feature_dim}"
    (
        train,
        train_targets,
        test,
        sample_submission,
        features,
        train_targets_nonscored,
    ) = load_data()
    tabnet_params = dict(
            feature_columns=None,
            num_classes=206,
            feature_dim=feature_dim,
            output_dim=feature_dim,
            num_features=len(features),
            num_decision_steps=1,
            relaxation_factor=1.5,
            sparsity_coefficient=1e-5,
            batch_momentum=0.98,
            virtual_batch_size=None,
            norm_type="group",
            num_groups=-1,
            multi_label=True,
        )
    _, _ = mlp_tf.run_mlp_tf_logger(
        train,
        test,
        train_targets,
        features,
        train_targets_nonscored,
        model_type="tabnet_class",
        str_condition=str_condition,
        tabnet_params=tabnet_params,
    )
    (
        train,
        train_targets,
        test,
        sample_submission,
        features,
        train_targets_nonscored,
    ) = load_data()
    tabnet_params = dict(
            feature_columns=None,
            num_classes=206,
            num_layers=2,
            feature_dim=feature_dim,
            output_dim=feature_dim,
            num_features=len(features),
            num_decision_steps=1,
            relaxation_factor=1.5,
            sparsity_coefficient=1e-5,
            batch_momentum=0.98,
            virtual_batch_size=None,
            norm_type="group",
            num_groups=-1,
            multi_label=True,
        )
    _, _ = mlp_tf.run_mlp_tf_logger(
        train,
        test,
        train_targets,
        features,
        train_targets_nonscored,
        model_type="stack_tabnet",
        str_condition=str_condition,
        tabnet_params=tabnet_params,
    )

for num_layers in [1, 2, 3, 4, 5]:
    str_condition = f"no feature_eng + num_layers={num_layers}"
    (
        train,
        train_targets,
        test,
        sample_submission,
        features,
        train_targets_nonscored,
    ) = load_data()
    tabnet_params = dict(
            feature_columns=None,
            num_classes=206,
            num_layers=num_layers,
            feature_dim=128,
            output_dim=64,
            num_features=len(features),
            num_decision_steps=1,
            relaxation_factor=1.5,
            sparsity_coefficient=1e-5,
            batch_momentum=0.98,
            virtual_batch_size=None,
            norm_type="group",
            num_groups=-1,
            multi_label=True,
        )
    _, _ = mlp_tf.run_mlp_tf_logger(
        train,
        test,
        train_targets,
        features,
        train_targets_nonscored,
        model_type="stack_tabnet",
        str_condition=str_condition,
        tabnet_params=tabnet_params,
    )

str_condition = "cliping"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_clipping(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "cliping min_clip=0.05, max_clip=0.95"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_clipping(train, test, min_clip=0.05, max_clip=0.95)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "fe_clipping_col"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
#display(np.max(train))
train, test = datasets.fe_clipping_col(train, test)
#display(np.max(train))
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "fe_clipping_col min_clip=0.05, max_clip=0.95"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
#display(np.max(train))
train, test = datasets.fe_clipping_col(train, test, min_clip=0.05, max_clip=0.95)
#display(np.max(train))
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "cliping → fe_stats"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_clipping(train, test)
train, test, features = datasets.fe_stats(train, test, flag_add=False)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "cliping → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_clipping(train, test)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "cliping → fe_stats → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_clipping(train, test)
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "no feature_eng + seeds=[5, 12]"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    seeds=[5, 12],
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "no feature_eng + p_min=0.001"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
    p_min=0.001,
)

str_condition = "no feature_eng + p_min=0.0001"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
    p_min=0.0001,
)

str_condition = "no feature_eng + p_min=0.000012"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
    p_min=0.000012,
)

str_condition = "feature_eng: RankGauss"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_quantile_transformer(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: Kmean"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_cluster(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: c_abs"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.c_abs(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: g_valid"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.g_valid(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: g_abs"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.g_abs(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: g_binary"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.g_binary(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: c_binary"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.c_binary(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: RankGauss → pca → variance_threshold"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_quantile_transformer(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=600, n_components_c=50, SEED=42
)
train, test, features = datasets.fe_variance_threshold(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: RankGauss → pca → variance_threshold → Kmean"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_quantile_transformer(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=600, n_components_c=50, SEED=42
)
train, test, features = datasets.fe_variance_threshold(train, test)
train, test, features = datasets.fe_cluster(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: RankGauss → pca → variance_threshold → Kmean → fe_stats flag_add=False → c_squared"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_quantile_transformer(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=600, n_components_c=50, SEED=42
)
train, test, features = datasets.fe_variance_threshold(train, test)
train, test, features = datasets.fe_cluster(train, test)
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: RankGauss → pca → variance_threshold → Kmean → scaling(RobustScaler)→ fe_stats flag_add=False → c_squared"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_quantile_transformer(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=600, n_components_c=50, SEED=42
)
train, test, features = datasets.fe_variance_threshold(train, test)
train, test, features = datasets.fe_cluster(train, test)
train, test, features = datasets.scaling(train, test)
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)


str_condition = "feature_eng: RankGauss → pca → variance_threshold → Kmean → fe_stats flag_add=False → c_squared → c_abs → scaling(RobustScaler)"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_quantile_transformer(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=600, n_components_c=50, SEED=42
)
train, test, features = datasets.fe_variance_threshold(train, test)
train, test, features = datasets.fe_cluster(train, test)
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.c_abs(train, test)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: scaling(RobustScaler)"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: scaling(StandardScaler)"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.scaling(train, test, scaler=StandardScaler())
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: fe_stats flag_add=True"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=True)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: fe_stats flag_add=False"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: fe_stats g only"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, params=["g"])
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: fe_stats c only"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, params=["c"])
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: fe_stats g,c"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, params=["g", "c"])
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: fe_stats gc only"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, params=["gc"])
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: scaling → fe_stats flag_add=True"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.scaling(
    train, test
)  # fe_statsの後にするとinfエラーが出て実行できない(infないんだけどなあ。。。)
train, test, features = datasets.fe_stats(train, test, flag_add=True)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: g_squared"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.g_squared(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: g_squared → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.g_squared(train, test)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: c_squared"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.c_squared(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: c_squared → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: fe_pca"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: fe_pca is_fit_train_only=True"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123, is_fit_train_only=True,
)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

# pcaの値確認
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train

# pcaの値確認
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123, is_fit_train_only=True,
)
train

str_condition = "feature_eng: fe_pca → scaling(RobustScaler)"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: fe_pca is_fit_train_only=True → scaling(RobustScaler)"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123, is_fit_train_only=True,
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "feature_eng: fe_pca → scaling(StandardScaler)"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test, scaler=StandardScaler())
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)


# + code_folding=[]
def split_gc(train, test):
    """g-,c-で特徴量分ける"""
    features_g = list(train.columns[2:4]) + [
        col for col in train.columns if "g-" in col
    ]
    features_c = list(train.columns[2:4]) + [
        col for col in train.columns if "c-" in col
    ]
    start_predictors_g = [s for s in mlp_tf.start_predictors if "g-" in s]
    start_predictors_c = [s for s in mlp_tf.start_predictors if "c-" in s]
    return (
        train[features_g],
        test[features_g],
        train[features_c],
        test[features_c],
        features_g,
        features_c,
        start_predictors_g,
        start_predictors_c,
    )


# -

str_condition = "g- feature only"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, _, _, features, _, start_predictors_g, _ = split_gc(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "g- feature only → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, _, _, features, _, start_predictors_g, _ = split_gc(train, test)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "c- feature only"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _, train, test, _, features, _, start_predictors_c = split_gc(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "c- feature only → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _, train, test, _, features, _, start_predictors_c = split_gc(train, test)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)


str_condition = "feature_eng: c_squared → scaling → fe_stats flag_add=True"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.scaling(train, test)
train, test, features = datasets.fe_stats(train, test, flag_add=True)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)


str_condition = "fe_stats flag_add=False → g_squared → c_squared → fe_pca → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.g_squared(train, test)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)


str_condition = (
    "fe_stats flag_add=False → c_squared → c_abs → g_valid → fe_pca → scaling"
)
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.c_abs(train, test)
train, test, features = datasets.g_valid(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)


str_condition = "fe_stats flag_add=False → c_squared → c_abs → g_valid → fe_pca is_fit_train_only=True → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.c_abs(train, test)
train, test, features = datasets.g_valid(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123, is_fit_train_only=True,
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "fe_stats flag_add=False → c_squared → fe_pca → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)
str_condition = "fe_stats flag_add=False → c_squared → c_abs → fe_pca → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.c_abs(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)
str_condition = "fe_stats flag_add=False → c_squared → c_abs → fe_pca → RankGauss"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.c_abs(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test = datasets.fe_quantile_transformer(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)
str_condition = "fe_stats flag_add=True → c_squared → fe_pca → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=True)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)
str_condition = "fe_ctl_mean"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_ctl_mean(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)
str_condition = "fe_ctl_mean gc_flg=g"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_ctl_mean(train, test, gc_flg="g")
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)
str_condition = "fe_ctl_mean gc_flg=c"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_ctl_mean(train, test, gc_flg="c")
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)
str_condition = "fe_ctl_mean is_mean=False"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_ctl_mean(train, test, is_mean=False)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)
str_condition = "fe_ctl_mean is_ratio=False"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_ctl_mean(train, test, is_ratio=False)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "fe_ctl_mean → fe_variance_threshold"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_ctl_mean(train, test)
train, test, features = datasets.fe_variance_threshold(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "fe_ctl_mean → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_ctl_mean(train, test)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "fe_ctl_mean → RankGauss"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_ctl_mean(train, test)
train, test = datasets.fe_quantile_transformer(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "RankGauss → fe_ctl_mean → fe_variance_threshold"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test = datasets.fe_quantile_transformer(train, test)
train, test, features = datasets.fe_ctl_mean(train, test)
train, test, features = datasets.fe_variance_threshold(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "fe_stats flag_add=False → c_squared → g_valid → fe_pca → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.g_valid(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = "fe_stats flag_add=False → c_squared → fe_ctl_mean → fe_pca → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.fe_ctl_mean(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)
str_condition = "fe_stats flag_add=False → c_squared → g_valid → fe_pca → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.g_valid(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = (
    "targets_nonscored + fe_stats flag_add=False → c_squared  → fe_pca → scaling"
)
(
    train,
    train_targets,
    test,
    sample_submission,
    train_targets_nonscored,
    train_drug,
) = datasets.load_orig_data()
train, train_targets, test, train_targets_nonscored = datasets.mapping_and_filter(
    train, train_targets, test, train_targets_nonscored, is_conat_nonscore=True
)
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

str_condition = (
    "targets_nonscored + fe_stats flag_add=False → c_squared  → fe_pca → scaling"
)
(
    train,
    train_targets,
    test,
    sample_submission,
    train_targets_nonscored,
    train_drug,
) = datasets.load_orig_data()
train, train_targets, test, train_targets_nonscored = datasets.mapping_and_filter(
    train, train_targets, test, train_targets_nonscored, is_conat_nonscore=True
)
train, test, features = datasets.fe_stats(train, test, flag_add=False)
train, test, features = datasets.c_squared(train, test)
train, test, features = datasets.fe_pca(
    train, test, n_components_g=70, n_components_c=10, SEED=123
)
train, test, features = datasets.scaling(train, test)
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="rs",
    str_condition=str_condition,
)
# +
str_condition = "EPOCHS = 200 + start_predictors → fe_stats flag_add=False → c_squared → c_abs → g_valid → fe_pca → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    train_targets_nonscored,
    train_drug,
) = datasets.load_orig_data()
train, train_targets, test, train_targets_nonscored = datasets.mapping_and_filter(
    train, train_targets, test, train_targets_nonscored, is_conat_nonscore=False
)

features_g, features_c = datasets.get_features_gc(
    train, top_feat_cols=mlp_tf.start_predictors
)

train, test, features = datasets.fe_stats(
    train, test, flag_add=False, features_g=features_g, features_c=features_c
)
train, test, features = datasets.c_squared(train, test, features_c=features_c)
train, test, features = datasets.c_abs(train, test, features_c=features_c)
train, test, features = datasets.g_valid(train, test, features_g=features_g)
train, test, features = datasets.fe_pca(
    train,
    test,
    n_components_g=70,
    n_components_c=10,
    SEED=123,
    features_g=features_g,
    features_c=features_c,
)
train, test, features = datasets.scaling(train, test)

mlp_tf.EPOCHS = 200
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="stack_tabnet",
    str_condition=str_condition,
)

# +
str_condition = "start_predictors → fe_stats flag_add=False → c_squared → c_abs → g_valid → fe_pca → scaling"
(
    train,
    train_targets,
    test,
    sample_submission,
    train_targets_nonscored,
    train_drug,
) = datasets.load_orig_data()
train, train_targets, test, train_targets_nonscored = datasets.mapping_and_filter(
    train, train_targets, test, train_targets_nonscored, is_conat_nonscore=False
)

features_g, features_c = datasets.get_features_gc(
    train, top_feat_cols=mlp_tf.start_predictors
)

train, test, features = datasets.fe_stats(
    train, test, flag_add=False, features_g=features_g, features_c=features_c
)
train, test, features = datasets.c_squared(train, test, features_c=features_c)
train, test, features = datasets.c_abs(train, test, features_c=features_c)
train, test, features = datasets.g_valid(train, test, features_g=features_g)
train, test, features = datasets.fe_pca(
    train,
    test,
    n_components_g=70,
    n_components_c=10,
    SEED=123,
    features_g=features_g,
    features_c=features_c,
)
train, test, features = datasets.scaling(train, test)

mlp_tf.EPOCHS = 80
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

# +
mlp_tf.LR = 0.001
mlp_tf.BATCH_SIZE = 256
mlp_tf.EPOCHS = 80

str_condition = "no feature_eng + batch_size=256"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

# +
mlp_tf.LR = 0.001
mlp_tf.BATCH_SIZE = 1024
mlp_tf.EPOCHS = 80

str_condition = "no feature_eng + batch_size=1024"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

# +
mlp_tf.LR = 0.001
mlp_tf.BATCH_SIZE = 64
mlp_tf.EPOCHS = 80

str_condition = "no feature_eng + batch_size=64"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _ = mlp_tf.run_mlp_tf_logger(b
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

# +
mlp_tf.LR = 0.001
mlp_tf.BATCH_SIZE = 32
mlp_tf.EPOCHS = 80

str_condition = "no feature_eng + batch_size=32"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

# +
mlp_tf.LR = 0.01
mlp_tf.BATCH_SIZE = 128
mlp_tf.EPOCHS = 80

str_condition = "no feature_eng + LR=0.01"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

# +
mlp_tf.LR = 0.01
mlp_tf.BATCH_SIZE = 128
mlp_tf.EPOCHS = 100

str_condition = "no feature_eng + LR=0.01 + epochs=100"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)

# +
mlp_tf.LR = 0.1
mlp_tf.BATCH_SIZE = 128
mlp_tf.EPOCHS = 80

str_condition = "no feature_eng + LR=0.1"
(
    train,
    train_targets,
    test,
    sample_submission,
    features,
    train_targets_nonscored,
) = load_data()
_, _ = mlp_tf.run_mlp_tf_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_type="4l",
    str_condition=str_condition,
)
# -

mlp_tf.LR = 0.001
mlp_tf.BATCH_SIZE = 128
mlp_tf.EPOCHS = 80



# +
def nelder_mead_weights_class(y_true: np.ndarray, oof_preds: list, method="L-BFGS-B"):
    """ネルダーミードでモデルのクラスごとのブレンド重み最適化
    ネルダーミード遅いから、L-BFGS-B をデフォルトの方法にする.重みクラス数文あるのでめちゃめちゃ時間かかる。。。"""
    from scipy.optimize import minimize

    def opt(ws, y_true, y_preds):
        y_pred = None
        for y_p in y_preds:
            if y_pred is None:
                y_pred = ws * y_p
            else:
                y_pred += ws * y_p
        return mlp_tf.mean_log_loss(y_true, y_pred)

    initial_weights = np.array([1.0 / y_true.shape[1]] * y_true.shape[1])
    if method in ["L-BFGS-B", "TNC", "COBYLA ", "SLSQP"]:
        # パラメータの範囲に制約のある方法で最適化
        # 1位の人は L-BFGS-B 使ってたので
        # https://www.kaggle.com/gogo827jz/optimise-blending-weights-with-bonus-0
        # パラメータの範囲は bounds に，最小値と最大値の対の系列を指定する必要あり
        # http://www.kamishima.net/mlmpyja/lr/optimization.html
        bnds = [(0, 1) for _ in range(y_true.shape[1])]
        result = minimize(
            opt,
            x0=initial_weights,
            args=(y_true, oof_preds),
            method=method,
            bounds=bnds,
        )
    else:
        result = minimize(
            opt, x0=initial_weights, args=(y_true, oof_preds), method=method
        )
    best_weights = result.x
    return best_weights


if "sig_id" in train_targets.columns:
    train_targets = train_targets.drop(["sig_id"], axis=1)
best_weights = nelder_mead_weights_class(train_targets.values,  [oof_pred, oof_pred * 2, oof_pred * 3])
best_weights
# -



# ## run_mlp_tf_blend_logger

# +
str_condition = "posi_ratio_class_weight"

(
    train,
    train_targets,
    test,
    sample_submission,
    train_targets_nonscored,
    train_drug,
) = datasets.load_orig_data()
train, train_targets, test, train_targets_nonscored = datasets.mapping_and_filter(
    train, train_targets, test, train_targets_nonscored
)
features = datasets.get_features(train)

mlp_tf.run_mlp_tf_blend_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_types=[
        "2l",
        "3l",
        "4l",
        "5l",
        "rs",
        "stack_tabnet",
        "tabnet_class",
    ],
    str_condition=str_condition,
    class_weight = datasets.fetch_posi_ratio_class_weight(train_targets),
)

# +
str_condition = "no feature_eng + posi_ratio_class_weight + targets_nonscored"

(
    train,
    train_targets,
    test,
    sample_submission,
    train_targets_nonscored,
    train_drug,
) = datasets.load_orig_data()
train, train_targets, test, train_targets_nonscored = datasets.mapping_and_filter(
    train, train_targets, test, train_targets_nonscored, is_conat_nonscore=True
)
features = datasets.get_features(train)

mlp_tf.run_mlp_tf_blend_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    model_types=[
        "2l",
        "3l",
        "4l",
        "5l",
        "rs",
        "stack_tabnet",
        "tabnet_class",
    ],
    str_condition=str_condition,
    class_weight = datasets.fetch_posi_ratio_class_weight(train_targets),
)

# +
for lr in [0.1, 0.01, 0.03, 0.0001, 0.00001]:
    mlp_tf.LR = lr
    str_condition = f"no feature_eng + learning rate={mlp_tf.LR}"
    (
        train,
        train_targets,
        test,
        sample_submission,
        features,
        train_targets_nonscored,
    ) = load_data()
    mlp_tf.run_mlp_tf_blend_logger(
        train,
        test,
        train_targets,
        features,
        train_targets_nonscored,
        str_condition=str_condition,
        model_types=[
            "2l",
            "3l",
            "4l",
            "5l",
            "rs",
            "3l_v2",
        ],
    )
    
mlp_tf.LR = 0.001

# +
for bs in [8, 32, 128, 512, 1024]:
    mlp_tf.BATCH_SIZE = bs
    str_condition = f"no feature_eng + mlp_tf.BATCH_SIZE={mlp_tf.BATCH_SIZE}"
    (
        train,
        train_targets,
        test,
        sample_submission,
        features,
        train_targets_nonscored,
    ) = load_data()
    mlp_tf.run_mlp_tf_blend_logger(
        train,
        test,
        train_targets,
        features,
        train_targets_nonscored,
        str_condition=str_condition,
        model_types=[
            "2l",
            "3l",
            "4l",
            "5l",
            "rs",
            "3l_v2",
        ],
    )
    
mlp_tf.BATCH_SIZE = 128

# +
from IPython.display import HTML

HTML(
    """
<button id="code-show-switch-btn">スクリプトを非表示にする</button>

<script>
var code_show = true;

function switch_display_setting() {
    var switch_btn = $("#code-show-switch-btn");
    if (code_show) {
        $("div.input").hide();
        code_show = false;
        switch_btn.text("スクリプトを表示する");
    }else {
        $("div.input").show();
        code_show = true;
        switch_btn.text("スクリプトを非表示にする");
    }
}

$("#code-show-switch-btn").click(switch_display_setting);
</script>
"""
)
