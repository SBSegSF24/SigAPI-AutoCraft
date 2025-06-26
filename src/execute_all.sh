#!/bin/bash

sigapi(){
    DATASET=$1
    python3 main.py --parallelize -d $DATASET --autocraft
}

for DATASET in ../Datasets/Completos/defensedroid_*.csv
do
    sigapi "$DATASET";
done
