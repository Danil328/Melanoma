#!/usr/bin/env bash

set -ex

cd ../src
dvc run -d ../data/train.csv \
        -o ../data/splits \
        python split_data.py --path ../data/train.csv \
                             --target_column target \
                             --n_folds 5 \
                             --holdout_size 0.1 \
                             --output ../data/splits
