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

# https://blog.amedama.jp/entry/sklearn-dummy-estimator  
# ### 説明変数の内容は使わず、主に目的変数の内容を代わりに使って、その名の通りダミーの結果を返す。特定のデータセットと評価指標を使ったときの、最低ラインの性能を確認するのに便利そう  
#
# <br>
#
# ### DummyClassifier は、その名の通りダミーの分類器となる。 使い方は一般的な scikit-learn の分類器と何ら違いはない。 違いがあるとすれば与えた教師データに含まれる説明変数を学習しないという点
#
# <br>
#
# ###  デフォルトでは、教師データに含まれる目的変数の確率分布を再現するように動作する
#
# <br>
#
# ### DummyClassifier は、返す値を色々とカスタマイズできる。 
# - 例えば、最頻値を常に返したいときはインスタンス化するときの strategy オプションに 'most_frequent' を指定する 
# - 任意の定数を返したいときは 'constant' を指定する
# - 目的変数の元の確率分布に依存せず、一様分布にしたいときは 'uniform' を指定

# https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
y[y != 1] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy='most_frequent', random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

y_pred_proba = clf.predict_proba(X_test)
y_pred_proba

from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)

clf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)



# +
import numpy as np
import pandas as pd

DATADIR = (
    r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\input\lish-moa"
)

dtype = {"cp_type": "category", "cp_dose": "category"}
index_col = "sig_id"

train_features = pd.read_csv(
    f"{DATADIR}/train_features.csv", dtype=dtype, index_col=index_col
)
X = train_features.select_dtypes("number")
Y_nonscored = pd.read_csv(
    f"{DATADIR}/train_targets_nonscored.csv", index_col=index_col
)
Y = pd.read_csv(f"{DATADIR}/train_targets_scored.csv", index_col=index_col)
groups = pd.read_csv(
    f"{DATADIR}/train_drug.csv", index_col=index_col, squeeze=True
)

columns = Y.columns

# +
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain

i = 0

clf = MultiOutputClassifier(DummyClassifier(random_state=i))
clf.fit(X, Y)
val_preds = clf.predict_proba(X_test)
val_preds = np.array(val_preds)[:,:,1].T  # take the positive class
val_preds
# -




