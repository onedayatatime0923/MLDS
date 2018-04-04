#!/bin/bash

for i in {1..100}
do
   python train.py  -lo loss_+'i' -ra ratio_+'i' -u 24 16 4 1
   echo "Welcome $i times"
done

