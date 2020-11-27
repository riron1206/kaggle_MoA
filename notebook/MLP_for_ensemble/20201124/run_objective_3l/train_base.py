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

# tensorflow2.0 + kerasでGPUメモリの使用量を抑える方法(最小限だけ使うように設定)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import sys
sys.path.append(r'C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\notebook\MLP_for_ensemble\20201124')
import moa_MLPs_funcs

params = dict(
    activation="relu",
    denses=[1159, 960, 1811],
    drop_rates=[
        0.4914099166744246,
        0.18817607797795838,
        0.12542057776853896,
        0.20175242230280122,
    ],
)
model_type = "3l_v2"

oof_score, Y_pred = moa_MLPs_funcs.train_and_evaluate(model_type=model_type, params=params)

oof_score
