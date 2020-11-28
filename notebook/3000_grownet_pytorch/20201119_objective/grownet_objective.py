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

# https://www.kaggle.com/tmhrkt/grownet-gradient-boosting-neural-networks

# !pwd
import sys
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
sys.executable

# GPU使えてるか確認
import torch
print(torch.cuda.is_available())

import grownet_funcs

# # test run
# - Kernel restart しないと前の結果が残るので注意！！！

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["batch_size"] = 16
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["batch_size"] = 1024
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["batch_size"] = 256
params["model"] = "MLP_2HL_weight_norm"
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["batch_size"] = 256
params["model"] = "MLP_2HL_weight_norm"
params["lr"] = 0.01
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["batch_size"] = 256
params["model"] = "MLP_2HL_weight_norm"
params["lr"] = 0.005
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["batch_size"] = 256
params["model"] = "MLP_2HL_weight_norm"
params["lr"] = 0.0001
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["batch_size"] = 256
params["model"] = "MLP_2HL_weight_norm"
params["lr"] = 0.001
params["epochs_per_stage"] = 3
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["batch_size"] = 256
params["model"] = "MLP_2HL_weight_norm"
params["lr"] = 0.001
params["epochs_per_stage"] = 1
params["boost_rate"] = 10.0
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["batch_size"] = 256
params["model"] = "MLP_2HL_weight_norm"
params["lr"] = 0.001
params["epochs_per_stage"] = 1
params["boost_rate"] = 1.0
params["hidden_size"] = 128
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["batch_size"] = 256
params["model"] = "MLP_2HL_weight_norm"
params["lr"] = 0.001
params["epochs_per_stage"] = 1
params["boost_rate"] = 1.0
params["hidden_size"] = 512
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["batch_size"] = 256
params["optimizer"] = "adabelief"
params["model"] = "MLP_2HL_weight_norm"
params["lr"] = 0.001
params["epochs_per_stage"] = 1
params["boost_rate"] = 1.0
params["hidden_size"] = 512
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["optimizer"] = "adabelief"
params["model"] = "MLP_2HL_weight_norm"
params["lr"] = 0.03
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["model"] = "MLP_2HL_weight_norm"
params["epochs_per_stage"] = 3
params["correct_epoch"] = 3
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["model"] = "MLP_2HL_weight_norm"
params["epochs_per_stage"] = 2
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["model"] = "MLP_2HL_weight_norm"
params["epochs_per_stage"] = 3
params["hidden_size"] = params["feat_d"] // 2
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["model"] = "MLP_2HL_weight_norm"
params["epochs_per_stage"] = 3
params["hidden_size"] = params["feat_d"] // 3
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["model"] = "MLP_2HL_weight_norm"
params["epochs_per_stage"] = 3
params["hidden_size"] = params["feat_d"]
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["model"] = "MLP_2HL_weight_norm"
params["epochs_per_stage"] = 3
params["weight_decay"] = 0.0
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 15
params["model"] = "MLP_2HL_weight_norm"
params["epochs_per_stage"] = 3
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 15
params["model"] = "MLP_2HL_weight_norm"
params["optimizer"] = "adabelief"
params["epochs_per_stage"] = 3
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 15
params["model"] = "MLP_2HL_weight_norm"
params["optimizer"] = "adabelief"
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 30
params["model"] = "MLP_2HL_weight_norm"
params["optimizer"] = "adabelief"
params["lr"] = 0.0001
score, y_pred = train_fn(n_seeds)

# test
from grownet_funcs import train_fn, params
n_seeds = 1
params["num_nets"] = 5
params["model"] = "MLP_2HL_leaky_relu"
params["epochs_per_stage"] = 3
score, y_pred = train_fn(n_seeds)







# # objective

# +
import optuna


def objective(trial):
    params = {
        "feat_d": len(grownet_funcs.feature_cols),
        "early_stopping_steps": 5,
        "n_folds": 5,
        "hidden_size": 512,
        "num_nets": 40,
        "epochs_per_stage": 1,  # Number of epochs to learn the Kth model. original: 1
        "correct_epoch": 1,  #  Number of epochs to correct the whole week models original: 1
        "model_order": "second",  # You could put "first" according to the original implemention, but error occurs. original: "second"
    }
    params["model"]   = trial.suggest_categorical("model", ["MLP_2HL",  "MLP_2HL_weight_norm"])
    params["batch_size"]   = trial.suggest_categorical("batch_size", [8, 16, 64, 256, 1024])
    params["lr"]           = trial.suggest_categorical("lr", [0.001, 0.01, 0.03])
    params["optimizer"]   = trial.suggest_categorical("optimizer", ["adam",  "adabelief"])
    params["weight_decay"] = trial.suggest_loguniform("weight_decay", 1e-07, 1e-3)
    #params["hidden_size"]  = trial.suggest_categorical("hidden_size", [16, 64, 128, 256, 512, 1024])
    params["boost_rate"]   = trial.suggest_categorical("boost_rate", [1.0, 5.0])  # ブースティング率. 修正ステップ中に自動的に調整される  # 0.5, 
    #params["num_nets"]     = trial.suggest_categorical("num_nets", [20, 40, 100])
    
    grownet_funcs.params = params
    score, y_pred = grownet_funcs.train_fn(n_seeds)
    
    return score


# +
# %%time

n_seeds = 1
n_trials = 100

study = optuna.create_study(
    study_name="study",
    #storage=f"sqlite:///study.db",
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=1),
)
study.optimize(objective, n_trials=n_trials)
study.trials_dataframe().to_csv(f"objective_history.csv", index=False)
with open(f"objective_best_params.txt", mode="w") as f:
    f.write(str(study.best_params))
print(f"\nstudy.best_params:\n{study.best_params}")
# -

# # run best param 

# +
# %%time

params = study.best_params

params["feat_d"] = len(grownet_funcs.feature_cols)
params["early_stopping_steps"] = 30
params["n_folds"] = 5
params["hidden_size"] = 512
params["num_nets"] = 40
params["epochs_per_stage"] = 1
params["correct_epoch"] = 1
params["model_order"] = "second"

grownet_funcs.params = params
score, y_pred = grownet_funcs.train_fn(n_seeds)
# -

score

path = r"Y_pred.pkl"
with open(path, 'rb') as f:
    Y_pred = pickle.load(f)
Y_pred

# # predict test

# +
train_features = pd.read_csv(
    f"{DATADIR}/train_features.csv"
    #, dtype=dtype, index_col=index_col
)
X = train_features.select_dtypes("number")

#test_features = pd.read_csv("../input/lish-moa/test_features.csv")
#sample_submission = pd.read_csv("../input/lish-moa/sample_submission.csv")
sample_submission = pd.read_csv(f"{DATADIR}/sample_submission.csv")
test_features = pd.read_csv(f"{DATADIR}/test_features.csv")

test = test_features.copy()
with open("./clipped_features.pkl", "rb") as f:
    clipped_features = pickle.load(f)
test[X.columns] = clipped_features.transform(test[X.columns])
test

# +
pd.set_option('display.max_columns', None)

x_test = test[feature_cols].values
test_ds = TestDataset(x_test)
test_loader = DataLoader(test_ds, batch_size=params["batch_size"], shuffle=False)

predictions = np.zeros((len(test), len(target_cols)))
for seed in tqdm(range(n_seeds)):
    seed_everything(seed)
    
    for fold in range(params["n_folds"]):
        net_ensemble = DynamicNet.from_file(
            f"./{fold}FOLD_{seed}_.pth", lambda stage: MLP_2HL.get_model(stage, params)
        )
        if device == "cuda":
            net_ensemble.to_cuda()
        net_ensemble.to_eval()

        preds = []
        with torch.no_grad():
            for data in test_loader:
                x = data["x"].to(device)
                _, pred = net_ensemble.forward(x)
                preds.append(pred.sigmoid().detach().cpu().numpy())
        predictions += np.concatenate(preds) / (params["n_folds"] * n_seeds)

sample_submission[target_cols] = predictions

sample_submission.loc[:, ["atp-sensitive_potassium_channel_antagonist", "erbb2_inhibitor"]] = 0.000012

test = test.set_index("sig_id")
sample_submission = sample_submission.set_index("sig_id")
sample_submission[test["cp_type"] == "ctl_vehicle"] = 0.0

sample_submission.to_csv("submission.csv")

display(sample_submission)
