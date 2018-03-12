
#train
python3 train.py -u 8 8 8 8 8 8 1 -o deep
python3 train.py -u 16 8 16 4 1 -o medium
python3 train.py -u 128 1 -o shallow

#plot
python3 plot.py
