

#train
python3 train_mnist.py -m mnist_deep -lo record/loss_mnist_d.npy -ao record/accu_mnist_d.npy
python3 train_mnist.py -m mnist_medium -lo record/loss_mnist_m.npy -ao record/accu_mnist_m.npy
python3 train_mnist.py -m mnist_shallow -lo record/loss_mnist_s.npy -ao record/accu_mnist_s.npy
#plot
python3 plot_mnist.py -lo pic/loss_mnist.png -ao pic/accu_mnist.png


