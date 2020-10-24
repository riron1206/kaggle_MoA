# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import glob
import pathlib
import joblib
import warnings
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# +
# local test
DATADIR = r"C:\Users\yokoi.shingo\my_task\MoA_Prediction\input\lish-moa"
MODELDIR = r"20201021_lgb_train\model"

## kaggle karnel
# DATADIR = "../input/lish-moa"
# MODELDIR = "../input/20201021_lgb/"

# 5foldにするとkaggle data setにupできない。1000ファイルまでしか上げれないみたいなので
N_SPLITS = 4

# +
test = pd.read_csv(f"{DATADIR}/test_features.csv")
train_targets_scored = pd.read_csv(f"{DATADIR}/train_targets_scored.csv")
submission = pd.read_csv(f"{DATADIR}/sample_submission.csv")

model_dirs = glob.glob(f"{MODELDIR}/*")


# + code_folding=[]
def preprocess(df):
    df = df.copy()
    # カテゴリ型のラベルを2値化
    df.loc[:, "cp_type"] = df.loc[:, "cp_type"].map({"trt_cp": 0, "ctl_vehicle": 1})
    df.loc[:, "cp_dose"] = df.loc[:, "cp_dose"].map({"D1": 0, "D2": 1})
    return df


def load_model(model_path="model/fold00.model"):
    return joblib.load(model_path)


# +
# Preprocessing
test = preprocess(test)

sub = submission.copy()
for m_dir in tqdm(model_dirs):
    y_preds = []
    for fold_id in range(N_SPLITS):
        model = load_model(f"{m_dir}/fold{str(fold_id).zfill(2)}.model")
        X_test = test.drop(["sig_id"], axis=1)
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_preds.append(y_pred)

    _preds = sum(y_preds) / len(y_preds)
    sub[pathlib.Path(m_dir).stem] = _preds

# Postprocessing: cp_typeが'ctl_vehicle'の行は予測値を0に設定
_cols = train_targets_scored.columns.to_list()[:]
_cols.remove("sig_id")
sub.loc[test["cp_type"] == 1, _cols] = 0
sub.to_csv("submission.csv", index=False)

print(sub.shape)
