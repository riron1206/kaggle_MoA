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
# ### DummyClassifier は、返す値を色々とカスタマイズできる。 例えば、最頻値を常に返したいときはインスタンス化するときの strategy オプションに 'most_frequent' を指定する 

# https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators
















