

#train
python3 train_mnist.py -u 256 256 256 10 -lo record/loss_mnist.npy -go record/grad_mnist.npy
#plot
python3 plot_mnist.py -lo pic/loss_mnist.png -go pic/grad_mnist.png


