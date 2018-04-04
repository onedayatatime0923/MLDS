

#train
python3 train_cifar.py -m cifar_deep -lo record/loss_cifar_d.npy -ao record/accu_cifar_d.npy
python3 train_cifar.py -m cifar_medium -lo record/loss_cifar_m.npy -ao record/accu_cifar_m.npy
python3 train_cifar.py -m cifar_shallow -lo record/loss_cifar_s.npy -ao record/accu_cifar_s.npy
#plot
python3 plot_cifar.py -lo pic/loss_cifar.png -ao pic/accu_cifar.png


