
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
OUTPUT_DIR= './data'
TENSORBOARD_DIR= './runs/acgan'

dm = DataManager(LATENT_DIM,DISCRIMINATOR_UPDATE_NUM,GENERATOR_UPDATE_NUM)
dm.tb_setting(TENSORBOARD_DIR)
#dm.get_anime_data('anime', i_path= '{}/faces'.format(INPUT_DIR), c_path= '{}/tags_clean.csv'.format(INPUT_DIR))
dm.get_extra_data('extra', i_path= '{}/extra_data/images'.format(INPUT_DIR), c_path= '{}/extra_data/tags.csv'.format(INPUT_DIR))
data_size, label_dim= dm.dataloader('train',['extra'] )
print('data_size: {}'.format(data_size))
print('label_dim: {}'.format(label_dim))

generator= Generator(LATENT_DIM+ label_dim, GENERATOR_HIDDEN_CHANNEL, data_size[0]).cuda()
discriminator= Discriminator( data_size[0], DISCRIMINATOR_HIDDEN_CHANNEL, label_dim).cuda()
optimizer= [generator.optimizer( lr=1E-4, betas= (0.5,0.999)),discriminator.optimizer( lr=1E-4, betas= (0.5,0.999))]
print(generator)
print(discriminator)

train_record=[]
for epoch in range(1,EPOCHS+1):
    train_record.append(dm.train('train', generator, discriminator, optimizer, epoch, print_every=3))
    dm.val(generator, discriminator, n=10, epoch= epoch, grid_path= '{}/grid.png'.format(OUTPUT_DIR))
torch.save(generator,'generator_acgan.pt')
torch.save(discriminator,'discriminator_acgan.pt')
#np.save('record/acgan_train_record.npy', np.array(train_record))
