import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
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
LAMBDA = 10

def calc_gradient_penalty_2(netD, real_data, fake_data, BATCH_SIZE):
	# print "real_data: ", real_data.size(), fake_data.size()

	alpha = torch.rand(BATCH_SIZE, 1)
	if torch.cuda.is_available():
		alpha = alpha.cuda()
	alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 64, 64)
	
	interpolates = alpha * real_data + ((1 - alpha) * fake_data)
	if torch.cuda.is_available():
		interpolates = interpolates.cuda()

	interpolates = autograd.Variable(interpolates, requires_grad=True)

	disc_interpolates = netD(interpolates)

	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
							  grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
								  disc_interpolates.size()),
							  create_graph=True, retain_graph=True, only_inputs=True)[0]
	gradients = gradients.view(gradients.size(0), -1)

	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
	return gradient_penalty


def calc_gradient_penalty(D, real_data, generated_data):
	batch_size = real_data.size()[0]

	# Calculate interpolation
	alpha = torch.rand(batch_size, 1, 1, 1)
	alpha = alpha.expand_as(real_data)
	if torch.cuda.is_available():
		alpha = alpha.cuda()
	interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
	interpolated = Variable(interpolated, requires_grad=True)
	if torch.cuda.is_available():
		interpolated = interpolated.cuda()

	# Calculate probability of interpolated examples
	prob_interpolated = D(interpolated)

	# Calculate gradients of probabilities with respect to examples
	gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
						   grad_outputs=torch.ones(prob_interpolated.size()).cuda() if torch.cuda.is_available() else torch.ones(
						   prob_interpolated.size()),
						   create_graph=True, retain_graph=True)[0]

	# Gradients have shape (batch_size, num_channels, img_width, img_height),
	# so flatten to easily take norm per example in batch
	gradients = gradients.view(batch_size, -1)
	#self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

	# Derivatives of the gradient close to 0 can cause problems because of
	# the square root, so manually calculate norm and add epsilon
	gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

	gradient_penalty = ((gradients_norm - 1) ** 2).mean() * LAMBDA
	# Return gradient penalty
	return gradient_penalty

def train(n_epochs, train_loader):
	
	rand_inputs = Variable(torch.randn(64,latent_dim, 1, 1),volatile=True)

	G = Generator()
	D = WGP_Discriminator()
	if torch.cuda.is_available():
		rand_inputs = rand_inputs.cuda()
		G.cuda()
		D.cuda()

	# setup optimizer
	#optimizerD = optim.RMSprop(D.parameters(), lr=5e-5)
	#optimizerG = optim.RMSprop(G.parameters(), lr=5e-5)
	optimizerD = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
	optimizerG = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))

	D_loss_list = []
	G_loss_list = []
	D_real_acc_list = []
	D_fake_acc_list = []

	print("START training...")

	for epoch in range(n_epochs):
		start = time.time()
		D_total_loss = 0.0
		G_total_loss = 0.0

		for batch_idx, (real_data, _) in enumerate(train_loader):
			D_loss_tmp = 0.0
			batch_size = len(real_data)
			real_data = to_var(real_data)
				
			# ================================================================== #
			#                      Train the discriminator                       #
			# ================================================================== #
			for _ in range(5):
				
				D.zero_grad()
				outputs_real = D(real_data)

				z = torch.randn(batch_size, latent_dim, 1, 1)
				z = to_var(z)
				fake_images = G(z)
				outputs_fake = D(fake_images.detach())

				# Backprop and optimize
				gradient_penalty = calc_gradient_penalty(D, real_data, fake_images)
				D_loss = -(torch.mean(outputs_real) - torch.mean(outputs_fake)) + gradient_penalty
				D_loss_tmp += D_loss.data[0]
				D_loss.backward()
				optimizerD.step()

			# ================================================================== #
			#                        Train the generator                         #
			# ================================================================== #
			
			# Compute loss with fake images
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
					epoch+1, (batch_idx+1) * len(real_data), len(train_loader.dataset),
					100. * batch_idx * len(real_data)/ len(train_loader.dataset),
					D_loss_tmp , G_loss.data[0],
					timeSince(start, (batch_idx+1)*len(real_data)/ len(train_loader.dataset))),end='')
	
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
								'./wgangp_outimgs/fig_%03d.jpg' %(epoch+1), nrow=8)
		
		torch.save(G.state_dict(), './saves/save_models/GP_Generator_%03d.pth'%(epoch+1))
		torch.save(D.state_dict(), './saves/save_models/GP_Discriminator_%03d.pth'%(epoch+1))

	with open('./saves/wgan_gp/D_loss.pkl', 'wb') as fp:
		pickle.dump(D_loss_list, fp)
	with open('./saves/wgan_gp/G_loss.pkl', 'wb') as fp:
		pickle.dump(G_loss_list, fp)



def main():	
	FACES_DIR = "../data/faces/"
	EXTRA_DIR = "../data/extra_data/images/"

	train_data = AnimeDataset(FACES_DIR, EXTRA_DIR)

	print("Read Data Done !!!")
	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
	print("Enter Train")
	train(300, train_loader)



if __name__ == '__main__':
	#parser = argparse.ArgumentParser(description='WGAN Example')
	#parser.add_argument('--train_path', help='training data directory', type=str)
	#args = parser.parse_args()
	main()

