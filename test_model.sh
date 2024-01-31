#!/bin/bash

# array of all train_ds
declare -a model_id=("gsd0.24" "gsd0.5" "gsd1" "gsd2")
declare -a test_ds=("I" "II" "III" "VI")
# batch size - change with l model to 6
bs=8
cf="$HOME/src/csu/yolov5"
yolodir=$(realpath $cf)

for m in "${model_id[@]}"
do
    floc="runs/motiur/$m/weights/best.pt"
    for t in "${test_ds[@]}"
        do
            logf="20240123_test/motiur-$m-$t.txt"
            # touch $logf
            python -u $yolodir/val.py --task test --save-txt --verbose --batch-size $bs --single-cls --save-conf --verbose --img 1280 --project 20240123_test --name motiur-$m-$t --data data/$t.yaml --weights $floc &> $logf
            var=$(grep "all.*" $logf)
            echo "last 2 metrics are mAP 50 and map50-95:$var" > $logf
        done
done        

