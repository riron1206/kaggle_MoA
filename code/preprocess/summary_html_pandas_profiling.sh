#!/bin/bash
PWDDIR=`pwd`

PY=/c/Users/yokoi.shingo/my_data/utils/python/summary_html_pandas_profiling/summary_html_pandas_profiling.py

DATA_DIR=../../input/lish-moa
OUT_DIR=../../data/eda
mkdir -p ${OUT_DIR}

conda activate py37

python ${PY} -o ${OUT_DIR} -i ${DATA_DIR}/train_features.csv
python ${PY} -o ${OUT_DIR} -i ${DATA_DIR}/train_targets_nonscored.csv
python ${PY} -o ${OUT_DIR} -i ${DATA_DIR}/train_targets_scored.csv
python ${PY} -o ${OUT_DIR} -i ${DATA_DIR}/test_features.csv
python ${PY} -o ${OUT_DIR} -i ${DATA_DIR}/sample_submission.csv
