"""codeディレクトリの関数実行用"""
import os
import gc
import pathlib
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir))

import params
import mlp_tf
from util import Logger
from datasets import *

# from tabnet_tf import *

warnings.filterwarnings("ignore")
logger = Logger("./")

# デバッグ用にデータ減らすか
DEBUG = False
# DEBUG = True
if DEBUG:
    mlp_tf.FOLDS = 2  # cvの数
    mlp_tf.EPOCHS = 2
    logger.info(f"### DEBUG: FOLDS:{mlp_tf.FOLDS},  EPOCHS:{mlp_tf.EPOCHS} ###")
else:
    logger.info(f"### HONBAN: FOLDS:{mlp_tf.FOLDS},  EPOCHS:{mlp_tf.EPOCHS} ###")
# logger.result("model_type\tcondition\toof_log_loss")


def run_train_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    logger=logger,
    model_dir="model",
    model_type="3l",
    seeds=[123],
    str_condition="",
    p_min=1e-15,
):
    """モデル作成してログファイルに結果書き込む"""
    try:
        # train time
        test_pred, oof_pred = mlp_tf.train_and_evaluate(
            train,
            test,
            train_targets,
            features,
            train_targets_nonscored,
            seeds=seeds,
            model_type=model_type,
            model_dir=model_dir,
        )
        oof_log_loss = mlp_tf.mean_log_loss_train_targets_oof(oof_pred, p_min=p_min)
        oof_log_loss = round(oof_log_loss, 7)
        logger.info(
            f"model_type:{model_type}, oof:{str(oof_log_loss)}"
        )  # general.logに文字列書き込み
        logger.result(
            f"{model_type}\t{str_condition}\t{str(oof_log_loss)}"
        )  # result.logに文字列書き込み
        return test_pred, oof_pred
    except Exception:
        traceback.print_exc()
        return 0.0, 0.0


def run_trains_logger(
    train,
    test,
    train_targets,
    features,
    train_targets_nonscored,
    logger=logger,
    str_condition="",
    model_types=["lr", "2l", "3l", "4l", "5l", "3l_v2", "3lWN"],
):
    """モデルブレンディングしてログファイルに結果書き込む"""
    try:
        test_preds = []
        oof_preds = []

        def _train(_model_type, _str_condition):
            test_pred, oof_pred = run_train_logger(
                train,
                test,
                train_targets,
                features,
                train_targets_nonscored,
                logger=logger,
                model_type=_model_type,
                str_condition=_str_condition,
            )
            test_preds.append(test_pred)
            oof_preds.append(oof_pred)

        for m_t in model_types:
            _train(m_t, str_condition)
        model_type = "-".join(model_types)

        mean_oof = np.average(oof_preds, axis=0)
        mean_oof = mean_oof / len(oof_preds)
        log_loss = mlp_tf.mean_log_loss_train_targets_oof(mean_oof)
        log_loss = round(log_loss, 7)
        logger.info(
            f"model_type:{model_type}, oof:{str(log_loss)}"
        )  # general.logに文字列書き込み
        logger.result(
            f"{model_type}\t{str_condition}:Mean blend\t{str(log_loss)}"
        )  # result.logに文字列書き込み

        mean_test = np.average(test_preds, axis=0)
        mean_test = mean_test / len(test_preds)
        sample_submission_pred = mlp_tf.submission_post_process(
            mean_test, test, out_csv="submission_trains_mean.csv"
        )

        # ------- Nelder-Mead で最適なブレンディングの重み見つける -------
        print("running Nelder-Mead...")
        _train_targets = pd.read_csv(f"{params.DATADIR}/train_targets_scored.csv")
        _train_targets.drop(["sig_id"], inplace=True, axis=1)
        best_weights = mlp_tf.nelder_mead_weights(_train_targets.values, oof_preds)
        wei_oof = None
        for wei, pre in zip(best_weights, oof_preds):
            if wei_oof is None:
                wei_oof = wei * pre
            else:
                wei_oof += wei * pre
        log_loss = mlp_tf.mean_log_loss_train_targets_oof(wei_oof)
        log_loss = round(log_loss, 7)
        logger.info(
            f"model_type:{model_type}, oof:{str(log_loss)}"
        )  # general.logに文字列書き込み
        logger.result(
            f"{model_type}\t{str_condition}:Nelder-Mead blend\t{str(log_loss)}"
        )  # result.logに文字列書き込み

        wei_test = None
        for wei, pre in zip(best_weights, test_preds):
            if wei_test is None:
                wei_test = wei * pre
            else:
                wei_test += wei * pre
        sample_submission_pred = mlp_tf.submission_post_process(
            wei_test, test, out_csv="submission_trains_nelder.csv"
        )
        # ----------------------------------------------------------------
    except Exception:
        traceback.print_exc()
