#!/usr/bin/env bash

set -ex

for file in "$1"/*.yaml
do
    echo "Download model -" ${file}
    ok_tasks download-model ${file}
done