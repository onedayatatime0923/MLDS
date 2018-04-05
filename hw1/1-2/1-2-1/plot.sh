

#train
python3 train.py  -u 64 10 -po record/para1_a.npy -m all_layer
python3 train.py  -u 64 10 -po record/para2_a.npy -m all_layer
python3 train.py  -u 64 10 -po record/para3_a.npy -m all_layer
python3 train.py  -u 64 10 -po record/para4_a.npy -m all_layer
python3 train.py  -u 64 10 -po record/para5_a.npy -m all_layer
python3 train.py  -u 64 10 -po record/para6_a.npy -m all_layer
python3 train.py  -u 64 10 -po record/para7_a.npy -m all_layer
python3 train.py  -u 64 10 -po record/para8_a.npy -m all_layer
python3 train.py  -u 64 10 -po record/para9_a.npy -m all_layer
python3 train.py  -u 64 10 -po record/para10_a.npy -m all_layer

python3 train.py  -u 64 10 -po record/para1_1.npy -m first_layer
python3 train.py  -u 64 10 -po record/para2_1.npy -m first_layer
python3 train.py  -u 64 10 -po record/para3_1.npy -m first_layer
python3 train.py  -u 64 10 -po record/para4_1.npy -m first_layer
python3 train.py  -u 64 10 -po record/para5_1.npy -m first_layer
python3 train.py  -u 64 10 -po record/para6_1.npy -m first_layer
python3 train.py  -u 64 10 -po record/para7_1.npy -m first_layer
python3 train.py  -u 64 10 -po record/para8_1.npy -m first_layer
python3 train.py  -u 64 10 -po record/para9_1.npy -m first_layer
python3 train.py  -u 64 10 -po record/para10_1.npy -m first_layer

#plot
python3 plot.py  -o pic/output_all_layer.png -m all_layer -i  record/para1_a.npy record/para2_a.npy record/para3_a.npy record/para4_a.npy record/para5_a.npy record/para6_a.npy record/para7_a.npy record/para8_a.npy record/para9_a.npy record/para10_a.npy

python3 plot.py  -o pic/output_first_layer.png -m first_layer -i  record/para1_1.npy record/para2_1.npy record/para3_1.npy record/para4_1.npy record/para5_1.npy record/para6_1.npy record/para7_1.npy record/para8_1.npy record/para9_1.npy record/para10_1.npy



