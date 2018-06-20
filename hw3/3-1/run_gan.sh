#!/bin/bash
wget -O model.pt  'https://www.dropbox.com/s/t6wwajks1cy71nn/model_generator_40.pt?dl=0'
python3 3-1.py -lat 128
