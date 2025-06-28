#!/bin/bash
sigapi(){
    DATASET=$1
    python3 main.py --parellelize -d $DATASET --autocraft
}

for DATASET in ../datasets/baseline/*.csv
do
    sigapi "$DATASET";
done
