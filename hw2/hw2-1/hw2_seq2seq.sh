
wget https://www.dropbox.com/s/5svul9zkphjprgx/encoder.pt?dl=1 -O encoder.pt
wget https://www.dropbox.com/s/ejoefpjiyxaccpr/decoder.pt?dl=1 -O decoder.pt
python3 test.py $1 $2
