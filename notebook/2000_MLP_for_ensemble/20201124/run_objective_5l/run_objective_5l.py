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

# +
import optuna


def objective(trial):
    model_type = "5l"
    params = {}
    params["activation"] = trial.suggest_categorical("activation", ["relu", "elu"])
    params["denses"] = trial.suggest_categorical(
        "denses",
        [
            [512, 448, 384, 320, 256],
            [1024, 896, 768, 640, 512],
            [2048, 1792, 1536, 1280, 1024],
            [2560, 2048, 1524, 1012, 780],
            [512, 512, 512, 512, 512],
            [1024, 1024, 1024, 1024, 1024],
            [1024, 768, 1280, 1536, 1792],
            [1024, 1280, 1536, 768, 1792],
            [512, 768, 1280, 1536, 1792],
            [512, 1536, 1280, 768, 512],
        ],
    )
    drop_rates = []
    for i in range(6):
        drop_rate = trial.suggest_uniform(f'drop_rate{i}', 0.2, 0.8)
        drop_rates.append(drop_rate)
    params["drop_rates"] = drop_rates
    print("-" * 100)
    print(f"params: {params}")

    oof_score, Y_pred = moa_MLPs_funcs.train_and_evaluate(model_type="5l", params=params)

    return oof_score


# +
# %%time

n_trials = 100
#n_trials = 50

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
