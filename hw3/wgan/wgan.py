import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch import optim
import torchvision
import time
import argparse
import skimage.io
import skimage
from models import *
from utils import *
import pickle
import sys
torch.manual_seed(424)
BATCH_SIZE = 128
latent_dim = 100


def train(n_epochs, train_loader):
	
	rand_inputs = Variable(torch.randn(25,latent_dim, 1, 1),volatile=True)

	G = Generator()
	D = Discriminator()
	if torch.cuda.is_available():
		rand_inputs = rand_inputs.cuda()
		G.cuda()
		D.cuda()

	# setup optimizer
	optimizerD = optim.RMSprop(D.parameters(), lr=5e-5)
	optimizerG = optim.RMSprop(G.parameters(), lr=5e-5)

	D_loss_list = []
	G_loss_list = []
	D_real_acc_list = []
	D_fake_acc_list = []

	print("START training...")

	for epoch in range(n_epochs):
		start = time.time()
		D_total_loss = 0.0
		G_total_loss = 0.0

		for batch_idx, (data, _) in enumerate(train_loader):
			D_loss_tmp = 0.0
			batch_size = len(data)
			data = to_var(data)
				
			# ================================================================== #
			#                      Train the discriminator                       #
			# ================================================================== #
			for _ in range(5):
				# Weight clipping
				for p in D.parameters():
					p.data.clamp_(-0.01, 0.01)
				D.zero_grad()

				# Compute Loss using real images
				outputs_real = D(data)

				# Compute Loss using fake images
				z = torch.randn(batch_size, latent_dim, 1, 1)
				z = to_var(z)
				fake_images = G(z)
				outputs_fake = D(fake_images.detach())


				# Backprop and optimize

				D_loss = -(torch.mean(outputs_real) - torch.mean(outputs_fake))
				D_loss_tmp += D_loss.data[0]
				D_loss.backward()
				optimizerD.step()

			# ================================================================== #
			#                        Train the generator                         #
			# ================================================================== #
			
			# Compute loss with fake images
			# We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
			G.zero_grad()
			z = torch.randn(batch_size, latent_dim, 1, 1)
			z = to_var(z)
			fake_images = G(z)
			outputs = D(fake_images)
			G_loss = -torch.mean(outputs)
			G_loss.backward()
			optimizerG.step()

			D_loss_tmp = D_loss_tmp/5
			D_total_loss += D_loss_tmp
			G_total_loss += G_loss.data[0]


			if batch_idx % 5 == 0:		
				print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]| D_Loss: {:.6f} , G_loss: {:.6f}| Time: {}  '.format(
					epoch+1, (batch_idx+1) * len(data), len(train_loader.dataset),
					100. * batch_idx * len(data)/ len(train_loader.dataset),
					D_loss_tmp , G_loss.data[0],
					timeSince(start, (batch_idx+1)*len(data)/ len(train_loader.dataset))),end='')
	
		print('\n====> Epoch: {} \nD_loss: {:.6f} \nG_loss: {:.6f}'.format(
			epoch+1, D_total_loss/len(train_loader),
			G_total_loss/len(train_loader)))
		print('-'*88)

		D_loss_list.append(D_total_loss/len(train_loader))
		G_loss_list.append(G_total_loss/len(train_loader))


		G.eval()
		rand_outputs = G(rand_inputs)
		G.train()
		torchvision.utils.save_image(rand_outputs.cpu().data,
								'./wgan_outimgs/fig_%03d.jpg' %(epoch+1), nrow=5)
		
		torch.save(G.state_dict(), './saves/save_models/Generator_%03d.pth'%(epoch+1))
		torch.save(D.state_dict(), './saves/save_models/Discriminator_%03d.pth'%(epoch+1))

	with open('./saves/wgan/D_loss.pkl', 'wb') as fp:
		pickle.dump(D_loss_list, fp)
	with open('./saves/wgan/G_loss.pkl', 'wb') as fp:
		pickle.dump(G_loss_list, fp)



def main():	
	FACES_DIR = "../data/faces/"
	EXTRA_DIR = "../data/extra_data/images/"

	train_data = AnimeDataset(FACES_DIR, EXTRA_DIR)

	print("Read Data Done !!!")
	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
	print("Enter Train")
	#train(150, train_loader)



if __name__ == '__main__':
	#parser = argparse.ArgumentParser(description='WGAN Example')
	#parser.add_argument('--train_path', help='training data directory', type=str)
	#args = parser.parse_args()
	main()

