#!/bin/bash
wget -O decoder_47.pt https://www.dropbox.com/s/wtt4tsuyfffkyox/decoder_47.pt?dl=1
wget -O encoder_47.pt https://www.dropbox.com/s/m0nbp6uq3z38plt/encoder_47.pt?dl=1
python3 model_seq2seq.py encoder_47.pt decoder_47.pt $1 $2
