#!/usr/bin/env bash

set -ex

cd ../src
python make_external_index_file.py \
        -p ../data/external_data/ISIS2019/ISIC_2019_Training_GroundTruth.csv \
        -p ../data/external_data/ISIS2018/ISIC2018_Task3_Training_GroundTruth.csv \
        -p ../data/external_data/ISIS2017/ISIC-2017_Training_Part3_GroundTruth.csv \
        -p ../data/external_data/ISIS2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv \
        -i image -i image -i image_id -i image \
        -t MEL -t MEL -t melanoma -t benign_malignant \
        -f ISIS2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input \
        -f ISIS2018/ISIC2018_Task3_Training_Input/ISIC2018_Task3_Training_Input \
        -f ISIS2017/ISIC-2017_Training_Data/ISIC-2017_Training_Data \
        -f ISIS2016/ISBI2016_ISIC_Part3_Training_Data \
        --output ../data/external_data/train.csv