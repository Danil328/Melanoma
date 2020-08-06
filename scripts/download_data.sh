#!/usr/bin/env bash

set -ex
cd ../data
kaggle competitions download siim-isic-melanoma-classification
unzip siim-isic-melanoma-classification.zip
rm siim-isic-melanoma-classification.zip
