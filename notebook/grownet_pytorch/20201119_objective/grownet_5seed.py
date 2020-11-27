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

from grownet_funcs import train_fn, params
n_seeds = 5
params["num_nets"] = 40
params["early_stopping_steps"] = 5
params["model"] = "MLP_2HL_weight_norm"
params["epochs_per_stage"] = 3
score, y_pred = train_fn(n_seeds)

# +
import pickle

path = r"Y_pred.pkl"
with open(path, 'rb') as f:
    Y_pred = pickle.load(f)
Y_pred
# -

# # predict test

# +
import pandas as pd
from grownet_funcs import *


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
        if params["model"] == "MLP_2HL_weight_norm":
            net_ensemble = DynamicNet.from_file(
                f"./{fold}FOLD_{seed}_.pth",
                lambda stage: MLP_2HL_weight_norm.get_model(stage, params),
            )
        elif params["model"] == "MLP_2HL_leaky_relu":
            net_ensemble = DynamicNet.from_file(
                f"./{fold}FOLD_{seed}_.pth",
                lambda stage: MLP_2HL_leaky_relu.get_model(stage, params),
            )
        else:
            net_ensemble = DynamicNet.from_file(
                f"./{fold}FOLD_{seed}_.pth",
                lambda stage: MLP_2HL.get_model(stage, params),
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
