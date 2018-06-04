
from util import DataManager,  Generator ,Discriminator
import torch
import numpy as np
assert DataManager and Generator and Discriminator and np


LATENT_DIM= 128
GENERATOR_UPDATE_NUM= 1
DISCRIMINATOR_UPDATE_NUM= 1
OUTPUT_DIR= './sample'
TENSORBOARD_DIR= './runs/acgan'

dm = DataManager(LATENT_DIM,DISCRIMINATOR_UPDATE_NUM,GENERATOR_UPDATE_NUM)
dm.tb_setting(TENSORBOARD_DIR)

generator= torch.load('model/generator_acgan.pt')
discriminator= torch.load('model/discriminator_acgan.pt')
print(generator)
print(discriminator)

dm.val(generator, discriminator, n=5, epoch= 0, grid_path= '{}/cgan.jpg'.format(OUTPUT_DIR))
