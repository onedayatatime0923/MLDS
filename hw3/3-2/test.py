
from util import DataManager,  Generator ,Discriminator
import torch
import numpy as np
import sys
assert DataManager and Generator and Discriminator and np


LATENT_DIM= 128
GENERATOR_UPDATE_NUM= 1
DISCRIMINATOR_UPDATE_NUM= 1
INPUT_PATH= sys.argv[1]
OUTPUT_PATH= './samples/cgan.jpg'
HAIR_PATH= 'hair.txt'
EYES_PATH= 'eyes.txt'

dm = DataManager(LATENT_DIM,DISCRIMINATOR_UPDATE_NUM,GENERATOR_UPDATE_NUM, HAIR_PATH, EYES_PATH)

generator= torch.load('model/generator_acgan.pt')
discriminator= torch.load('model/discriminator_acgan.pt')
print(generator)
print(discriminator)

dm.test(generator, discriminator, INPUT_PATH, OUTPUT_PATH)
