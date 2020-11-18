# データディレクトリ
# DATADIR = "../input/lish-moa"
DATADIR = (
    r"C:\Users\81908\jupyter_notebook\poetry_work\tfgpu\01_MoA_compe\input\lish-moa"
)
# マルチラベルStratifiedKFoldのディレクトリ
# ※StratifiedKFoldとは、「CVの際に、目的変数のラベルの比率が揃うように訓練データと検証データを分ける」分割方法
# ITERATIVE_STRATIFICATION = '../input/iterativestratification'
ITERATIVE_STRATIFICATION = r"C:\Users\81908\Git\iterative-stratification"

# adabelief_tf0.1.0
# ADABELIEF_TF = "../input/adabelief-tf010-pip"
ADABELIEF_TF = r"C:\Users\81908\Git\Adabelief-Optimizer\pypi_packages\adabelief_tf0.1.0"

FOLDS = 5  # cvの数 5で固定

# MLPのパラメータ
EPOCHS = 1000  # 80
BATCH_SIZE = 16  # 128
LR = 0.001
VERBOSE = 0  # 0だと学習履歴出さない

# lgmのパラメータ
