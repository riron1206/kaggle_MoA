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

1024*6

# +
import optuna


def objective(trial):
    model_type = "stacked_tabnet"
    
    params = dict(
        epsilon=1e-05,
        feature_columns=None,  # データセットのTensorflow特徴列
        virtual_batch_size=None,  # 仮想バッチサイズ。全体のバッチサイズは virtual_batch_size の整数倍じゃないとだめらしい
        num_layers=2,  # 重ねるTabNetsの数
        num_decision_steps=1,  # decision stepsの数
        norm_type="batch",  # 正規化のタイプ
        num_groups=-1,  # group normarizaionのグループの数。よくわからん
        batch_momentum=0.9, # 仮想バッチのMomentum。よくわからん
        relaxation_factor=1.2, 
        sparsity_coefficient=trial.suggest_categorical("lambda_sparse", [0, 0.000001, 0.0001, 0.001, 0.01, 0.1]),  # 論文の探索範囲。sparsity正則化
    )
    # feature_dim must be larger than output dim
    # feature_dim must be a list of length `num_layers`
    params["feature_dim"] = trial.suggest_categorical("Na", [1024, 1536, 2048, 2560])  # feature transformation block  # , 3072, 4096, 6144
    params["output_dim"] = trial.suggest_categorical("Nd", [32, 64, 128])  # decision step

    print("-" * 100)
    print(f"params: {params}")

    oof_score, Y_pred = moa_MLPs_funcs.train_and_evaluate(
        model_type=model_type, params=params
    )

    return oof_score


# +
# %%time

import warnings
warnings.filterwarnings('ignore')

#n_trials = 100
n_trials = 30
#n_trials = 3

study = optuna.create_study(
    study_name="study",
    storage=f"sqlite:///study.db",
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=1),
)
study.optimize(objective, n_trials=n_trials)
study.trials_dataframe().to_csv(f"objective_history.csv", index=False)
with open(f"objective_best_params.txt", mode="w") as f:
    f.write(str(study.best_params))
print(f"\nstudy.best_params:\n{study.best_params}")
print(f"\nstudy.best_value:\n{study.best_value}")
