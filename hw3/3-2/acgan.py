
from util import DataManager,  Generator ,Discriminator_Acgan
import torch
import numpy as np
assert DataManager and Generator and Discriminator_Acgan


BATCH_SIZE=  128
EPOCHS= 200
LATENT_DIM= 128
GENERATOR_HIDDEN_CHANNEL = 128
DISCRIMINATOR_HIDDEN_CHANNEL = 128
GENERATOR_UPDATE_NUM= 1
DISCRIMINATOR_UPDATE_NUM= 1
OUTPUT_DIR= './data/acgan'
TENSORBOARD_DIR= './runs/acgan'

dm = DataManager(LATENT_DIM,DISCRIMINATOR_UPDATE_NUM,GENERATOR_UPDATE_NUM)
dm.tb_setting(TENSORBOARD_DIR)
data_size, label_dim=dm.get_data('train', i_path= './data/extra_data/images', c_path= './data/extra_data/tags.csv',batch_size= BATCH_SIZE, shuffle=True)
print('data_size: {}'.format(data_size))
print('label_dim: {}'.format(label_dim))

generator= Generator(LATENT_DIM+ label_dim, GENERATOR_HIDDEN_CHANNEL, data_size[0]).cuda()
discriminator= Discriminator_Acgan( data_size[0], DISCRIMINATOR_HIDDEN_CHANNEL, label_dim).cuda()
optimizer= [generator.optimizer( lr=1E-4, betas= (0.5,0.999)),discriminator.optimizer( lr=1E-4, betas= (0.5,0.999))]
print(generator)
print(discriminator)
dm.tb_graph((generator,discriminator), LATENT_DIM)

train_record=[]
for epoch in range(1,EPOCHS+1):
    train_record.append(dm.train_acgan('train', generator, discriminator, optimizer, epoch, print_every=5))
    dm.val_acgan(generator, discriminator, epoch= epoch, n=10, path=OUTPUT_DIR)
torch.save(generator,'generator_acgan.pt')
torch.save(discriminator,'discriminator_acgan.pt')
np.save('record/acgan_train_record.npy', np.array(train_record))
