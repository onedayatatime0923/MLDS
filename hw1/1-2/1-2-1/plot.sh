

#train
python3 train.py  -u 64 10 -po record/para1.npy -m all_layer
python3 train.py  -u 64 10 -po record/para2.npy -m all_layer
python3 train.py  -u 64 10 -po record/para3.npy -m all_layer
python3 train.py  -u 64 10 -po record/para4.npy -m all_layer
python3 train.py  -u 64 10 -po record/para5.npy -m all_layer
python3 train.py  -u 64 10 -po record/para6.npy -m all_layer
python3 train.py  -u 64 10 -po record/para7.npy -m all_layer
python3 train.py  -u 64 10 -po record/para8.npy -m all_layer
python3 train.py  -u 64 10 -po record/para9.npy -m all_layer
python3 train.py  -u 64 10 -po record/para10.npy -m all_layer

#plot

#plot
python3 plot.py  -o pic/output.png -i  record/para1.npy record/para2.npy record/para3.npy record/para4.npy record/para5.npy record/para6.npy record/para7.npy record/para8.npy record/para9.npy record/para10.npy -m all_layer



