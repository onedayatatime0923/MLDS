import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


# for GAN

class Generator(nn.Module):
	def __init__(self, ngf=64):
		super(Generator, self).__init__()
		self.decoder = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d( 100, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(inplace=True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(inplace=True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(inplace=True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(inplace=True)
			# state size. (ngf) x 32 x 32
		)
		self.output = nn.Sequential(
			nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
			nn.Tanh()
			# state size. (3) x 64 x 64
		)

	def forward(self, x):
		hidden = self.decoder(x)
		output = self.output(hidden)/2.0+0.5
		return output
	
class Discriminator(nn.Module):
	def __init__(self, ndf=64):
		super(Discriminator, self).__init__()
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(ndf),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True)
			# state size. (ndf*8) x 4 x 4
		)
		self.output = nn.Sequential(
			nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		hidden = self.main(x)
		output = self.output(hidden)

		return output.view(-1, 1).squeeze(1)


if __name__ == '__main__':
	G = ACGenerator()
	D = ACDiscriminator()
	print(G)
	print(D)

