"""Util funcs"""
import datetime
import logging
import os
import pathlib
import numpy as np


class Logger:
    def __init__(self, output_dir=None):
        self.general_logger = logging.getLogger("general")
        self.result_logger = logging.getLogger("result")
        stream_handler = logging.StreamHandler()

        # ディレクトリ指定無ければカレントディレクトリにログファイル出す
        output_dir = pathlib.Path.cwd() if output_dir is None else output_dir
        file_general_handler = logging.FileHandler(
            os.path.join(output_dir, "general.log")
        )
        file_result_handler = logging.FileHandler(
            os.path.join(output_dir, "result.log")
        )

        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        """時刻をつけてコンソールとgeneral.log（ログファイル）に文字列書き込み"""
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    def result_scores(self, run_name, scores):
        """
        計算結果をコンソールとresult.log（cv結果用ログファイル）に書き込み
        parms: run_name: 実行したcvの名前
        parms: scores: cv scoreのリスト。result.logには平均値も書く
        """
        dic = dict()
        dic["name"] = run_name
        dic["score"] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f"score{i}"] = score
        self.result(self.to_ltsv(dic))

    def result(self, message):
        """コンソールとresult.logに文字列書き込み"""
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        """コンソールとresult.logに辞書データ書き込み"""
        self.result(self.to_ltsv(dic))

    def now_string(self):
        """時刻返すだけ"""
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_ltsv(self, dic):
        """辞書を文字列に変えるだけ"""
        return "\t".join(["{}:{}".format(key, value) for key, value in dic.items()])
