#!/bin/bash

# array of all train_ds
declare -a train_ds=("1b" "1c" "2a" "1bc" "12c" "12ab" "12ac" "12abc")
declare -a model_size=("l" "m" "s")
declare -a test_ds=("I" "II" "III" "VI")
# batch size - change with l model to 6
bs=8
cf="$HOME/src/csu/yolov5"
yolodir=$(realpath $cf)

for d in "${train_ds[@]}"
do
    for m in "${model_size[@]}"
    do
    if [ "$m" = "l" ];
    then
        bs=4
    else
        bs=8 
    fi
        python $yolodir/train.py --data data/$d.yaml --project 20240123 --name $m-$d --img 1280 --weights yolov5$m.pt --batch-size $bs --epochs 300 --noplots
        for t in "${test_ds[@]}"
        do
            logf="20240123_test/$m-$d-$t.txt"
            # touch $logf
            python -u val.py --task test --save-txt --verbose --batch-size $bs --single-cls --save-conf --verbose --img 1280 --project 20240123_test --name $m-$d-$t --data data/$t.yaml --weights 20240123/$m-$d/weights/best.pt &> $logf
            var=$(grep "all.*" $logf)
            echo "last 2 metrics are mAP 50 and map50-95: $var" > $logf
        done
    done        
done

