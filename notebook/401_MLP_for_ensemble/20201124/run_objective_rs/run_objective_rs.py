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

sys.path.append(
    r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\notebook\MLP_for_ensemble\20201124"
)
import moa_MLPs_funcs

# +
import optuna


def objective(trial):
    model_type = "rs"
    params = {}

    activations = []
    for i in range(8):
        activation = trial.suggest_categorical(f"activation{i}", ["relu", "elu", "selu"])
        activations.append(activation)
    params["activations"] = activations

    denses = []
    for i in range(6):
        #dense = trial.suggest_int(f"dense{i}", 128, 1024)
        dense = trial.suggest_int(f"dense{i}", 256, 1024)  # oldの結果より
        denses.append(dense)
    params["denses"] = denses

    drop_rates = []
    for i in range(2):
        #drop_rate = trial.suggest_uniform(f"drop_rate{i}", 0.1, 0.9)
        drop_rate = trial.suggest_uniform(f"drop_rate{i}", 0.5, 0.9)  # oldの結果より
        drop_rates.append(drop_rate)
    params["drop_rates"] = drop_rates

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

#n_trials = 400
n_trials = 100
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
