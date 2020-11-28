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

params = {
    "batch_momentum": 0.95,
    "feature_dim": 512,
    "norm_type": "batch",
    "num_decision_steps": 1,
    #"num_layers": 2,
    "num_layers": 5,
}
model_type = "stacked_tabnet"

oof_score, Y_pred = moa_MLPs_funcs.train_and_evaluate(model_type=model_type, params=params)

oof_score
