#!/usr/bin/env bash

set -ex

cd ../src

directory=../data/tars
if [[ -d ${directory} ]]; then rm -Rf ${directory}; fi
mkdir ${directory}

python make_tar_file.py --path ../data/jpeg ../data/test ../data/tfrecords ../data/train --output ../data/tars/melanoma --step 1000

hdfs dfs -put ${directory}/* /antispam/kaggle/melanoma_classification/data/