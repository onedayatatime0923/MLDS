import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import time
import math
import numpy as np
import pandas as pd
import scipy.misc
import os
from PIL import Image

def read_anime(filepath):
	print(filepath)
	images = []	
	img_path_list = os.listdir(filepath)
	img_path_list.sort()

	for i, file in enumerate(img_path_list):
		#img = scipy.misc.imread(os.path.join(filepath, file))
		img = Image.open(os.path.join(filepath, file)).resize((64, 64), Image.ANTIALIAS)
		images.append(np.array(img))

	images = np.array(images)/255.0
	images = images.transpose(0,3,1,2)
	return images

def read_extra(filepath,flip=False):
	images = []	

	print(filepath)
	img_path_list = os.listdir(filepath)
	img_path_list.sort()
	for i, file in enumerate(img_path_list):
		img = scipy.misc.imread(os.path.join(filepath, file))
		images.append(img)
		if(flip):
			images.append(np.fliplr(img))

	images = np.array(images)/255.0
	images = images.transpose(0,3,1,2)
	return images

def to_var(x):
	x = Variable(x)
	if torch.cuda.is_available():
		x = x.cuda()
	return x

def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

class AnimeDataset(Dataset):
	"""docstring for MyDataset"""
	def __init__(self, faces_filepath, extra_filepath):
		#self.train_data = read_image(train_filepath)
		#self.test_data = read_image(test_filepath)		

		self.anime = read_anime(faces_filepath)
		self.extra = read_extra(extra_filepath)
		self.images = np.concatenate((self.anime, self.extra), axis=0)
		print(self.images.shape)
		self.images = torch.FloatTensor(self.images)

	def __getitem__(self, index):
		data = self.images[index]

		return data, data

	def __len__(self):
		return len(self.images)


if __name__ == '__main__':
	FACES_DIR = "../data/faces/"
	EXTRA_DIR = "../data/extra_data/images/"
	#TRAIN_CSVDIR = "./hw4_data/train.csv"
	#TEST_CSVDIR = "./hw4_data/test.csv"

	all_imgs = read_anime(FACES_DIR)	#(33431, 3, 96, 96)
	extra_imgs = read_extra(EXTRA_DIR)	#(36740, 3, 64, 64)
	print(all_imgs.shape)
	print(extra_imgs.shape)



