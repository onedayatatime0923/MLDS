
from util import DataManager,  Generator ,Discriminator
import torch
import numpy as np
assert DataManager and Generator and Discriminator and np


BATCH_SIZE=  128
EPOCHS= 200
LATENT_DIM= 128
GENERATOR_HIDDEN_CHANNEL = 128
DISCRIMINATOR_HIDDEN_CHANNEL = 128
GENERATOR_UPDATE_NUM= 1
DISCRIMINATOR_UPDATE_NUM= 1
INPUT_DIR= '../data'
OUTPUT_DIR= './sample'
TENSORBOARD_DIR= './runs/acgan'

dm = DataManager(LATENT_DIM,DISCRIMINATOR_UPDATE_NUM,GENERATOR_UPDATE_NUM)
dm.tb_setting(TENSORBOARD_DIR)
dm.get_extra_data('extra', i_path= '{}/extra_data/images'.format(INPUT_DIR), c_path= '{}/extra_data/tags.csv'.format(INPUT_DIR))
dm.get_anime_data('anime', i_path= '{}/faces'.format(INPUT_DIR), c_path= '{}/tags_clean.csv'.format(INPUT_DIR))
data_size, label_dim= dm.dataloader('train',['extra', 'anime'] )
print('data_size: {}'.format(data_size))
print('label_dim: {}'.format(label_dim))

generator= torch.load('model/generator_acgan.pt')
discriminator= torch.load('model/discriminator_acgan.pt')
print(generator)
print(discriminator)

dm.val(generator, discriminator, n=10, epoch= 0, grid_path= '{}/grid.png'.format(OUTPUT_DIR))
