#!/bin/bash
PWDDIR=`pwd`

PY=/c/Users/yokoi.shingo/my_data/utils/python/tsv_summary/tsv_summary.py

DATA_DIR=../../input/lish-moa
OUT_DIR=../../data/eda
mkdir -p ${OUT_DIR}

conda activate py37

python ${PY} \
        ${DATA_DIR}/train_features.csv \
        ${DATA_DIR}/train_targets_nonscored.csv \
        ${DATA_DIR}/train_targets_scored.csv \
        ${DATA_DIR}/test_features.csv \
        ${DATA_DIR}/sample_submission.csv \
        -s , \
        -o ${OUT_DIR}/summary.tsv
