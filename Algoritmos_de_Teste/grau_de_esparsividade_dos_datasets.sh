#!/bin/bash

for i in datasets/*
do 
    python3 grau_de_esparsividade.py --dataset $i
done
