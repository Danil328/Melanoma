#!/usr/bin/env bash

set -ex

cd ../data
rm -r external_data
mkdir external_data
cd external_data

mkdir ISIS2019
mkdir ISIS2018
mkdir ISIS2017
mkdir ISIS2016

cd ISIS2019
kaggle datasets download andrewmvd/isic-2019
unzip isic-2019.zip
rm isic-2019.zip

cd ../ISIS2018
kaggle datasets download shonenkov/isic2018
unzip isic2018.zip
rm isic2018.zip

cd ../ISIS2017
kaggle datasets download shonenkov/isic2017
unzip isic2017.zip
rm isic2017.zip

cd ../ISIS2016
kaggle datasets download shonenkov/isic2016
unzip isic2016.zip
rm isic2016.zip