import scipy.misc
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os
from torch.autograd import Variable
import torch.nn as nn
import argparse 
import torch.optim as optim
import torchvision
import random , time , math
import matplotlib.pyplot as plt 

parser =  argparse.ArgumentParser(description='dcgan model') 
parser.add_argument('-lat',type=int,dest='latent_size',required=True) 
args = parser.parse_args() 

class imagedataset(Dataset):
    def __init__(self,img1,img2):
        self.img1 = img1
        self.img2 = img2 
    def __getitem__(self,i):    
        l = self.img1.size()[0]
        if i < l : x = self.img1[i]
        else : x = self.img2[i-l]
        return x
    def __len__(self):
        return self.img1.shape[0] + self.img2.shape[0] 

def get_data(path,num,name,batch_size,shuffle=False):
    data = os.listdir(path)
    data.sort()
    print('num=',len(data))
    input()
    #data = data[:num]
    arr = []
    for x in data:
        print(os.path.join(path,x))
        im = Image.open(os.path.join(path,x))
        im = im.resize((64,64), Image.ANTIALIAS)
        im = np.array(im)
        #print(im.shape)
        #input()
        im = (im/127.5)-1
        #print(im)
        #input()
        arr.append(im)
    arr = np.array(arr)
    print('saving data ...')
    np.save(name+'.npy',arr)
    arr = torch.FloatTensor(arr)
    dataset = imagedataset(arr,arr)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

class discriminator(nn.Module):
    def __init__(self,nc,ngf,ndf,latent_size):
        super(discriminator,self).__init__()
        self.nc = nc #3
        self.ngf = ngf #64
        self.ndf = ndf #64
        self.latent_size = latent_size 
        #(3,64,64)
        #self.e1 = nn.Conv2d(3,32,4,2,1, bias=True) #(32,32,32)
        #self.bn1 = nn.BatchNorm2d(32)

        self.e2 = nn.Conv2d(3,64,4,2,1, bias=True) #(64,32,32)
        self.bn2 = nn.BatchNorm2d(64)
 
        self.e3 = nn.Conv2d(64,128,4,2,1, bias=True) #(128,16,16)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.e4 = nn.Conv2d(128,256,4,2,1, bias=True) #(256,8,8)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.e5 = nn.Conv2d(256,512,4,2,1, bias=True) #(512,1,1)
        self.bn5 = nn.BatchNorm2d(512)

        self.e6 = nn.Conv2d(512,1,4,1,0, bias=True) #(1,1,1)

        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2,inplace=True)
    
    def forward(self,x):
        x = x.permute(0,3,1,2)
        #print('x size: ',x.size())
        #input()
        #h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(x)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h6 = self.sigmoid(self.e6(h5))
 
        h6 = h6.view(-1,1).squeeze(1)
        return h6 

class generator(nn.Module):
    def __init__(self,nc,ngf,ndf,latent_size):
        super(generator,self).__init__()
        self.nc = nc #3
        self.ngf = ngf #64
        self.ndf = ndf #64
        self.latent_size = latent_size 

        self.up1 = nn.ConvTranspose2d(self.latent_size,512,4,1,0, bias=False) # (512,4,4)
        self.bn6 = nn.BatchNorm2d(512)

        self.up2 = nn.ConvTranspose2d(512,256,4,2,1, bias=False) # (256,8,8)
        self.bn7 = nn.BatchNorm2d(256)

        self.up3 = nn.ConvTranspose2d(256,128,4,2,1, bias=False) # (128,16,16)
        self.bn8 = nn.BatchNorm2d(128)

        self.up4 = nn.ConvTranspose2d(128,64,4,2,1, bias=False) # (64,32,32)
        self.bn9 = nn.BatchNorm2d(64)
 
        self.up5 = nn.ConvTranspose2d(64,3,4,2,1, bias=False) # (3,64,64)
        #self.bn10 = nn.BatchNorm2d(32)
 
        #self.up6 = nn.ConvTranspose2d(32,3,4,2,1, bias=False) # (3,64,64)
        #self.bn11 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self,h0):
        #h0 = self.relu(self.d1(x)) # (1024)
        #h0 = h0.view(-1,1024,1,1)
        h0 = h0.view(-1,self.latent_size,1,1)
        h1 = self.relu(self.bn6(self.up1(h0)))
        h2 = self.relu(self.bn7(self.up2(h1)))
        h3 = self.relu(self.bn8(self.up3(h2)))
        h4 = self.relu(self.bn9(self.up4(h3)))
        h5 = self.tanh(self.up5(h4))
        #h5 = self.relu(self.bn10(self.up5(h4)))
        #h6 = self.tanh(self.up6(h5))
        return h5.permute(0,2,3,1) 
 
net_D = discriminator(nc=3, ngf=64, ndf=64, latent_size = args.latent_size ).cuda()
net_G = generator(nc=3, ngf=64, ndf=64, latent_size = args.latent_size ).cuda() 
#print(net_D)
#print('-'*50)
#print(net_G)
#print('-'*50) 
optimizerD = optim.Adam(net_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(net_G.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()


                                    
def rand_faces(generator):
    generator.eval()    
    #z = np.random.normal(0, 1, (25, args.latent_size))
    #z = Variable(torch.FloatTensor(z), volatile=True)
    z = Variable(torch.randn((25,args.latent_size)),volatile=True)
    z = z.cuda()
    recon = generator(z).cpu().data.numpy() 
    #print(recon)
    #input()
    recon = (recon+1)*127.5 
    recon = recon.astype(np.uint8)
    save_imgs(recon)

def save_imgs(gen_imgs):
    r, c = 5, 5 
    fig, axs = plt.subplots(r, c)
    cnt = 0 
    #print(gen_imgs)
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("samples/gan_original.png")
    plt.close()

torch.manual_seed(424)
model = torch.load('model/dcgan/model_generator_40.pt')
rand_faces(model)





























































































































































