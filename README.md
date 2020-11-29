# MoAコンペのコード
- https://www.kaggle.com/c/lish-moa

## コンペ概要
- 薬の作用機序(MoA)の応答を予測する
    - MoA. mechanism of action: 薬物が生体に何らかの効果を及ぼす仕組み、メカニズム。特定の分子の生物学的活性。要はお薬が病気の原因に届くまでの流れのこと。
        - 例. ロピニロールという薬は、主に線条体のドパミンD2受容体を直接刺激し、ドパミン様に働き抗パーキンソン病効果を示す
        - 参考: https://medical.kyowakirin.co.jp/neuro/hrp-tape/009.html

- テーブルデータ
    - 約5,000種類の薬を使った実験データ
        - 1行1実験
    - 1つの薬について7行ある
        - 6通りの実験条件(cp_time列, cp_dose列の組み合わせ) + 薬投与なし の7行
    - 薬の種類を識別する情報なし
        - KfoldでCV作ると同種の薬のデータが混じってリークする。薬の種類でGroupKFoldしたいができない
    - 特徴量はヒトの遺伝子発現や細胞生存率の数値データ
    - ※遺伝子発現（発現）. Gene expression：遺伝情報に基づいてタンパク質が合成されること（DNA→（転写）→RNA→（翻訳）→タンパク質のサイクルをセントラルドグマという）。また発現される量（発現量）のことを発現ということもある。
        - 列名が「g-」の特徴量(g-0からg-771の772列)がヒト遺伝子の発現値。1列1列がALKやPI3Kなど各遺伝子についての発現値のはず。
            - 発現値の種類は不明。-5-5の値なのでIC50ではなさそう
            - 「このデータは、100種類の細胞タイプのプール内の薬剤に対するヒト細胞の反応を同時に（同じサンプル内で）測定する新技術に基づいています」と書いてるからマイクロアレイとか次世代シーケンサーでとった値？
    - ※細胞生存率. Cell viability: 生細胞数/全細胞数のパーセンテージ。
        - 参考: https://www.abcam.co.jp/protocols/counting-cells-using-a-haemocytometer
        - 列名が「c-」の特徴量(c-0からc-99の100列)が細胞生存率。1列1列が皮膚がんの細胞や肺がんの細胞など各細胞についてのパーセント値のはず。
            - -10-5の値をとるので、「測定した細胞生存率-100」で規格化した値と思う。値が3なら3%細胞死んでることになるので薬が効いてるってこと？値が-10なら10%細胞増えてるので薬が悪影響与えてる？
            - 細胞名いろいろあるみたい: http://www.jhsf.or.jp/bank/CellNameJ.html
        - 癌細胞らしい
            - https://www.kaggle.com/c/lish-moa/discussion/190693
    - ラベルは206種の薬の作用機序(MoA)の応答（0/1）
        - 1列1クラス。クラス名が薬の名前。なので206クラスある
        - 薬は 5α還元酵素阻害剤、11-β-HSD1阻害剤 など
        - 非常に不均衡（0のラベルが大半で1のラベルが非常に少ない）

- マルチラベル分類問題

- 評価指標は各クラスのlog_lossの平均値


- コンペ概要日本語訳: https://www.kaggle.com/furuhatakazuki/japanese-moa-introduction

-------------------------------------
### 作業ログ
- https://github.com/riron1206/kaggle_MoA/issues/2
- https://trello.com/b/lHvX528J/kagglemoa

-------------------------------------
### 最終submitはチームでのアンサンブル
- https://www.kaggle.com/yxohrxn/votingclassifier-fit
- https://www.kaggle.com/yxohrxn/votingclassifier-predict

-------------------------------------
### 自分が作ったモデル
#### 2クラス分類のLightGBM
- cv: 0.01608
- https://www.kaggle.com/anonamename/moa-lightgbm

#### LGBMClassifier + ClassifierChain
- cv: 0.01661
- https://www.kaggle.com/anonamename/moa-lgbmclassifier-classifierchain

#### XGBClassifier + ClassifierChain
- cv: 0.01621, auc: 0.5581
- https://www.kaggle.com/anonamename/moa-xgbclassifier-classifierchain
    
#### Self-Stacking + 2クラス分類のXGBoost
- cv: 0.01603, auc: 0.7559
- ※第1段階の予測値を第2段階の学習の追加特徴量とし、クラス間の関係性を学習させる。陽性ラベル(=1)が多いクラスを最初に学習してoof出し(第1段階目)、そのoofを特徴量に追加して残りの陽性ラベル少ないクラス学習
- 学習に9時間以上かかったのでローカルで実行した結果をDatasetsにuploadした: https://www.kaggle.com/anonamename/moaselfstackingxgboost
- 学習コード: https://www.kaggle.com/anonamename/moa-self-stacking-xgboost?scriptVersionId=47884299
- 予測コード: https://www.kaggle.com/anonamename/moa-self-stacking-xgboost-calibrate?scriptVersionId=47996842 のIn[20],In[21],In[22]

#### cuml + 2クラス分類のSVM
- cv: 0.01625
- ※rapidsのSVMは1seed分しか保存できない。kaggleは出力ファイルを20GBまでしか保存できないため。5seed分作るためnotebook分けた
- https://www.kaggle.com/anonamename/moa-rapids-svm-seed-y-pred
- https://www.kaggle.com/anonamename/moa-rapids-svm-seed01
- https://www.kaggle.com/anonamename/fork-of-moa-rapids-svm-seed23
- https://www.kaggle.com/anonamename/fork-of-moa-rapids-svm-seed4
    
#### 遺伝的アルゴリズム + cuml + KNN
- cv: 0.01909（np.clip(oof,0.0005,0.999) + ctl行=0 にしたら cv: 0.01865）
- ※特徴量をスケーリングする重みを遺伝的アルゴリズムで計算し、cumlのKNNでマルチラベル分類する
- https://www.kaggle.com/anonamename/moa-genetic-algorithm-cuml-knn

#### DummyClassifier + MultiOutputClassifier
- cv: 0.02105
- ※DummyClassifierは特徴量を学習データに使わず、ラベルの確率分布を再現するルールベースのモデル。ランダムに出力してもこれくらいの精度になるらしい
- https://www.kaggle.com/anonamename/moa-dummyclassifier-multioutputclassifier

#### MLP(層の数変えた5種類のMLPのアンサンブル)
- cv: 0.01718, LB: 0.01861
- 学習コード: https://www.kaggle.com/anonamename/moa-dnn-feature-engineering-20201023-re-cv
- 予測コード: https://www.kaggle.com/anonamename/submission-moa-dnn-moa-code-20201104-v2?scriptVersionId=46198226
- パラメーターチューニングしたローカルでの実行結果: https://www.kaggle.com/anonamename/mlp-for-ensemble
    - 5l   : cv: 0.01611, auc: 0.64835
    - 4l   : cv: 0.01591, auc: 0.66564
    - 3l_v2: cv: 0.01576, auc: 0.68225
    - 2l   : cv: 0.01572, auc: 0.67610
    - rs   : cv: 0.01588, auc: 0.67808

#### StackedTabNet
- cv: , auc: 
- ※決定木(GBMT)をDNNで模倣したモデル。Encode(FC=>BN=>GLU(ゲート付き線形変換) + transformer。特徴量をEncodeしてtransformerに入れるのを繰り返す構造。RNNっぽい（時系列っぽく）何回もtransformerに入れる
- https://www.kaggle.com/anonamename/moa-stackedtabnet-fit
- パラメーターチューニングしたローカルでの実行結果: https://www.kaggle.com/anonamename/mlp-for-ensemble
        - StackedTabNet: cv: 0.015609, auc: 0.68547
- 自分のハイスコアモデル
    - ローカルでの実行notebook: https://github.com/riron1206/kaggle_MoA/blob/master/notebook/2000_MLP_for_ensemble/kaggle_upload/moa_MLPs_StackedTabNet_only.ipynb

#### GrowNet
- cv: 0.01589
- ※1,2層程度の浅いMLPを弱モデルとしてブースティング。浅いMLPを1epoch学習→最終層の出力を特徴量列に追加→浅いMLPを1epoch学習→… を繰り返す
- https://www.kaggle.com/anonamename/moa-grownet


-------------------------------------

### git 更新は以下のコマンドで出来た
```bash
$ git add --all
$ git commit -m "hogehoge"
$ git push origin HEAD:master
```
