

#train
python3 train.py -u 8 8 8 8 8 8 1 -f 1 -lo record/loss_d.npy -po record/pred_d.npy
python3 train.py -u 16 8 16 4 1   -f 1 -lo record/loss_m.npy -po record/pred_m.npy
python3 train.py -u 128 1         -f 1 -lo record/loss_s.npy -po record/pred_s.npy

#plot
python3 plot.py -f 1 -lo pic/loss1.png -po pic/pred1.png


#train
#python3 train.py -u 8 8 8 8 8 8 1 -f 2 -lo record/loss_d.npy -po record/pred_d.npy
#python3 train.py -u 16 8 16 4 1   -f 2 -lo record/loss_m.npy -po record/pred_m.npy
#python3 train.py -u 128 1         -f 2 -lo record/loss_s.npy -po record/pred_s.npy

#plot
#python3 plot.py -f 2 -lo pic/loss2.png -po pic/pred2.png
