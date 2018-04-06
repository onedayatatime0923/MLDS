#!/bin/bash
python3 train.py -b 64 -p weights/64
python3 train.py -b 1024 -p weights/1024
python3 interpolation.py
python3 plot.py