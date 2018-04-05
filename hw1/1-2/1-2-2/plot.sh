

#train
python3 train.py -u 256 256 256 10 -lo record/loss_mnist.npy -go record/grad_mnist.npy
#plot
python3 plot.py -o pic/output.png



