#!/bin/bash

#clear screen 
clear 
cd src
python3 train_classification.py --batchsize 32 --dataset 10 --learningRate 0.001 --epochs 20 --log "../logs/log.txt" --numPoints 1024
