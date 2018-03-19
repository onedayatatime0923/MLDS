

#train
python3 train.py  -u 512 512 10 -po record/para1.npy
python3 train.py  -u 512 512 10 -po record/para2.npy
python3 train.py  -u 512 512 10 -po record/para3.npy
python3 train.py  -u 512 512 10 -po record/para4.npy
python3 train.py  -u 512 512 10 -po record/para5.npy

#plot
python3 plot.py  -o pic/output.png -i  record/para1.npy record/para2.npy record/para3.npy record/para4.npy record/para5.npy



