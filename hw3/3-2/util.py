
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time, os, math
from tensorboardX import SummaryWriter 
import matplotlib.pyplot as plt
assert torch and nn and Variable and F and Dataset and DataLoader and SummaryWriter
assert time and np


class DataManager():
    def __init__(self,latent_dim=0, discriminator_update_num=0, generator_update_num=0):
        self.data={}
        self.hair= Color()
        self.eyes= Color()
        self.discriminator_update_num= discriminator_update_num
        self.generator_update_num= generator_update_num
        self.latent_dim= latent_dim
        self.writer= None
    def tb_setting(self, path):
        for f in os.listdir(path): 
            os.remove('{}/{}'.format(path,f))
        self.writer = SummaryWriter(path)
    def tb_graph(self, model, input_shape):
        if isinstance(input_shape, tuple):
            dummy_input= Variable( torch.rand(1, *input_shape).cuda())
        elif isinstance(input_shape, int):
            dummy_input= Variable( torch.rand(1, input_shape).cuda())
        else: raise ValueError('Wrong input_shape')
        self.writer.add_graph(nn.Sequential(*model), (dummy_input, ))
    def get_anime_data(self,name, i_path, c_path):
        x=[]
        file_len = len([file for file in os.listdir(i_path) if file.endswith('.jpg')])
        for i in range(file_len):
            im=Image.open('{}/{}.jpg'.format(i_path,i)).resize( (64, 64), Image.BILINEAR )
            x.append(np.array(im,dtype=np.uint8).transpose((2,0,1)))
            print('\rreading {} image...{}'.format(name,len(x)),end='')
        print('\rreading {} image...finish'.format(name))

        y=[]
        with open(c_path, 'r') as f:
            for line in f:
                data= line.replace('\n','').split(',',1)[1].split('\t')
                hair_c, eyes_c= -1, -1
                for c in data:
                    if 'hair' in c:
                        hair_c= self.hair.addColor(c.split(':',1)[0])
                for c in data:
                    if 'eyes' in c:
                        eyes_c= self.eyes.addColor(c.split(':',1)[0])
                y.append(np.array([hair_c,eyes_c],dtype=np.uint8))
                print('\rreading {} class...{}'.format(name,len(y)),end='')
        print('\rreading {} class...finish'.format(name))
        x=np.array(x)
        y=np.array(y)
        self.data[name]=[x,y]
    def get_extra_data(self,name, i_path, c_path):
        x=[]
        file_len = len([file for file in os.listdir(i_path) if file.endswith('.jpg')])
        for i in range(file_len):
            x.append(np.array(Image.open('{}/{}.jpg'.format(i_path,i)),dtype=np.uint8).transpose((2,0,1)))
            print('\rreading {} image...{}'.format(name,len(x)),end='')

        print('\rreading {} image...finish'.format(name))

        y=[]
        with open(c_path, 'r') as f:
            for line in f:
                data= line.replace('\n','').split(',',1)[1].split(' ')
                hair_c, eyes_c= self.hair.addColor(data[0]), self.eyes.addColor(data[2])
                y.append(np.array([hair_c,eyes_c],dtype=np.uint8))
                print('\rreading {} class...{}'.format(name,len(y)),end='')
        print('\rreading {} class...finish'.format(name))
        x=np.array(x)
        y=np.array(y)
        self.data[name]=[x,y]
    def dataloader(self, name, in_names, batch_size= 128, shuffle=True, flip=False):
        '''
        print(self.data['anime'][0].shape)
        print(self.data['anime'][1].shape)
        print(self.data['extra'][0].shape)
        print(self.data['extra'][1].shape)
        '''
        x= np.concatenate([self.data[i][0] for i in in_names], 0)
        y= np.concatenate([self.data[i][1] for i in in_names], 0)
        self.data[name]=DataLoader(FaceDataset(x, y, self.hair.n_colors, self.eyes.n_colors, flip= flip),batch_size=batch_size, shuffle=shuffle)
        return x.shape[1:], self.hair.n_colors+ self.eyes.n_colors
    def train(self,name, generator, discriminator, optimizer, epoch, print_every=1):
        start= time.time()
        generator.train()
        discriminator.train()
        
        generator_optimizer= optimizer[0]
        discriminator_optimizer= optimizer[1]
        
        criterion= nn.BCELoss()
        total_loss= [0,0,0]     # G, D, C
        batch_loss= [0,0,0]     # G, D, C
        total_accu= [0,0,0]     # G, D_real, D_fake
        batch_accu= [0,0,0]     # G, D_real, D_fake
        
        data_size= len(self.data[name].dataset)
        for j, (i, c) in enumerate(self.data[name]):
            batch_index=j+1
            origin_i = Variable(i).cuda()
            origin_c = Variable(c).cuda()
            # update discriminator
            for k in range(self.discriminator_update_num):
                hair_index= torch.rand(len(i),1).random_(0,self.hair.n_colors).long()
                latent_hair= torch.zeros(len(i), self.hair.n_colors).scatter_(1,hair_index,1)
                eyes_index= torch.rand(len(i),1).random_(0,self.eyes.n_colors).long()
                latent_eyes= torch.zeros(len(i), self.eyes.n_colors).scatter_(1,eyes_index,1)
                latent_c = torch.cat((latent_hair, latent_eyes),1)
                latent = Variable(torch.cat((torch.randn(len(i),self.latent_dim),latent_c),1).cuda())
                fake_i, fake_c= discriminator(generator(latent))
                real_i, real_c= discriminator(origin_i)
                zero= Variable( torch.rand(len(i),1)*0.3).cuda()
                one= Variable( torch.rand(len(i),1)*0.5 + 0.7).cuda()
                loss_fake_i= criterion( fake_i, zero)
                loss_real_i= criterion( real_i, one)
                loss_fake_c= criterion( fake_c, Variable(latent_c.cuda()))
                loss_real_c= criterion( real_c, origin_c)
                loss= (loss_fake_i + loss_fake_c + loss_real_i + loss_real_c)
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()
                batch_loss[1]+= float(loss_fake_i)+ float( loss_real_i)
                batch_loss[2]+= float(loss_fake_c)+ float( loss_real_c)
                batch_accu[1]+= int(torch.sum(real_i>0.5))
                batch_accu[2]+= int(torch.sum(fake_i<0.5))
                #print(float(loss))

            # update generator
            for k in range(self.generator_update_num):
                hair_index= torch.rand(len(i),1).random_(0,self.hair.n_colors).long()
                latent_hair= torch.zeros(len(i), self.hair.n_colors).scatter_(1,hair_index,1)
                eyes_index= torch.rand(len(i),1).random_(0,self.eyes.n_colors).long()
                latent_eyes= torch.zeros(len(i), self.eyes.n_colors).scatter_(1,eyes_index,1)
                latent_c = torch.cat((latent_hair, latent_eyes),1)
                latent = Variable(torch.cat((torch.randn(len(i),self.latent_dim),latent_c),1).cuda())
                fake_i, fake_c= discriminator(generator(latent))
                one= Variable( torch.rand(len(i),1)*0.5 + 0.7).cuda()
                loss_fake_i= criterion( fake_i, one)
                loss_fake_c= criterion( fake_c, Variable(latent_c.cuda()))
                loss= (loss_fake_i + loss_fake_c )
                #loss=  torch.mean(-torch.log(discriminator(generator(batch_x))))
                generator_optimizer.zero_grad()
                loss.backward()
                generator_optimizer.step()
                batch_loss[0]+= float(loss_fake_i)
                batch_loss[2]+= float(loss_fake_c)
                batch_accu[0]+= int(torch.sum(fake_i>0.5))
                #print(float(loss))

            if batch_index% print_every == 0:
                total_loss[0]+= batch_loss[0]/ (self.generator_update_num ) if (self.generator_update_num!=0) else 0
                total_loss[1]+= batch_loss[1]/ (self.discriminator_update_num*2 ) if (self.discriminator_update_num!=0) else 0
                total_loss[2]+= batch_loss[2]/ (self.discriminator_update_num*2 + self.generator_update_num )
                total_accu[0]+= batch_accu[0]/ (self.generator_update_num ) if (self.generator_update_num!=0) else 0
                total_accu[1]+= batch_accu[1]/ (self.discriminator_update_num ) if (self.discriminator_update_num!=0) else 0
                total_accu[2]+= batch_accu[2]/ (self.discriminator_update_num ) if (self.discriminator_update_num!=0) else 0
                print('\rEpoch {} | [{}/{} ({:.0f}%)] | Loss G: {:.4f} D: {:.4f} C: {:.4f} | Accu G: {:.1f}% D_real: {:.1f}% D_fake: {:.1f}% | Time: {}'.format(
                                epoch , batch_index*len(i), data_size, 
                                100. * batch_index*len(i)/ data_size,
                                batch_loss[0]/ (self.generator_update_num *print_every) if (self.generator_update_num!=0) else 0,
                                batch_loss[1]/ (self.discriminator_update_num*2 *print_every)if (self.discriminator_update_num!=0) else 0,
                                batch_loss[2]/ ((self.discriminator_update_num*2 + self.generator_update_num )* print_every),
                                100. * (batch_accu[0]/ (self.generator_update_num* print_every ))/ len(i) if (self.generator_update_num!=0) else 0,
                                100. * (batch_accu[1]/ (self.discriminator_update_num* print_every ))/ len(i) if (self.discriminator_update_num!=0) else 0,
                                100. * (batch_accu[2]/ (self.discriminator_update_num* print_every ))/ len(i) if (self.discriminator_update_num!=0) else 0,
                                self.timeSince(start, batch_index*len(i)/ data_size)),end='')
                batch_loss= [0,0,0]     # G, D, C
                batch_accu= [0,0,0]     # G, D_real, D_fake
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Total Loss G : {:.6f} D : {:.6f} C : {:.6f} | Total Accu G: {:.2f}% D_real: {:.2f}% D_fake: {:.2f}% | Time: {}  '.format(
                        epoch , data_size, data_size, 100. ,
                        float(total_loss[0])/batch_index,float(total_loss[1])/batch_index, float(total_loss[2])/ batch_index,
                        100.* float(total_accu[0])/ data_size, 100.* float(total_accu[1])/ data_size, 100.* float(total_accu[2])/ data_size,
                        self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('generator loss', float(total_loss[0])/ batch_index, epoch)
            self.writer.add_scalar('discriminator loss', float(total_loss[1])/ batch_index, epoch)
            self.writer.add_scalar('classification loss', float(total_loss[2])/ batch_index, epoch)
        print('-'*80)
        return total_loss[0]/ batch_index, total_loss[1]/ batch_index, total_loss[2]/ batch_index
    def val(self, generator, discriminator, n=20, hair_c=[0,1,2], eyes_c=[0,1], epoch=None, dir_path=None, grid_path= None):
        generator.eval()
        discriminator.eval()
        
        x= torch.randn(n,self.latent_dim)
        predict= []
        c_no= torch.FloatTensor([[ 0 for i in range(self.hair.n_colors + self.eyes.n_colors)]]).repeat(n,1)
        latent_no= Variable(torch.cat((x, c_no),1).cuda())
        predict.extend(generator(latent_no).cpu().data.unsqueeze(1))
        #for l in range(self.hair.n_colors + self.eyes.n_colors):
        for h in hair_c:
            for e in eyes_c:
                c_yes= torch.FloatTensor([[ int( i == h) for i in range(self.hair.n_colors)] + [ int( i == e) for i in range(self.eyes.n_colors)]]).repeat(n,1)
                latent_yes= Variable(torch.cat((x, c_yes),1).cuda())
                predict.extend(generator(latent_yes).cpu().data.unsqueeze(1))
        predict= torch.cat(predict,0)

        if self.writer!=None:
            self.writer.add_image('sample image result', torchvision.utils.make_grid(predict, normalize=True, range=(-1,1), nrow= n), epoch)
        if dir_path != None: self.write(predict,dir_path,'gan')
        if grid_path != None: self.plot_grid(torchvision.utils.make_grid((predict*127.5)+127.5, nrow= n), grid_path)
    def visualize_latent_space(self, name, encoder, path):
        class_0=[]
        class_1=[]
        for i, (x, c) in enumerate(self.data[name]):
            batch_x= Variable(x).cuda()
            mean_x, _ = encoder(batch_x)
            # predict
            c=c.squeeze(1)
            class_0_index= torch.nonzero(1-c).cuda().squeeze(1)
            class_0.extend(torch.index_select(mean_x.data, 0, class_0_index).unsqueeze(1))
            class_1_index= torch.nonzero(c).cuda().squeeze(1)
            class_1.extend(torch.index_select(mean_x.data, 0, class_1_index).unsqueeze(1))
        data=torch.cat(class_0 +class_1,0)
        print('PCA data reduct...',end='')
        data=PCA(n_components=4).fit_transform(data)
        print('\rTSNE data reduct...',end='')
        print('                       \r',end='')
        data= TSNE(n_components=2, random_state=23).fit_transform(data)
        plt.figure()
        plt.scatter(data[:len(class_0),0], data[:len(class_0),1], s= 5, c='r')
        plt.scatter(data[len(class_0):,0], data[len(class_0):,1], s= 5, c='b')
        plt.savefig(path)
    def write(self,data,path,mode):
        data=data.numpy()
        #print(output.shape)
        for i in range(data.shape[0]):
            im=(data[i].transpose((1,2,0))*127.5)+127.5
            im=im.astype(np.uint8)
            image = Image.fromarray(im,'RGB')
            image.save('{}/{:0>4}.png'.format(path,i))
    def plot_grid(self,im,path):
        im=im.numpy().transpose((1,2,0))
        #print(output.shape)
        im=im.astype(np.uint8)
        image = Image.fromarray(im,'RGB')
        image.save('{}'.format(path))
    def plot_record(self,data, path, title, color='b'):
        x=np.array(range(1,len(data)+1))
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(x,data[:,0],'b')
        plt.title(title[0])
        plt.subplot(2,1,2)
        plt.plot(x,data[:,1],'b')
        plt.title(title[1])
        plt.savefig(path)
    def timeSince(self,since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))
    def asMinutes(self,s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size ):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( input_size, hidden_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(True),
            # state size. (hidden_size*8) x 4 x 4
            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            # state size. (hidden_size*4) x 8 x 8
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            # state size. (hidden_size*2) x 16 x 16
            nn.ConvTranspose2d(hidden_size * 2,     hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            # state size. (hidden_size) x 32 x 32
            nn.ConvTranspose2d(    hidden_size, output_size, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ( output_size) x 64 x 64
        )
    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.main(x)
        return x 
    def make_layers(self, input_channel, cfg,  batch_norm=False):
        #cfg = [(64,2), (64,2)]
        layers = []
        in_channels = input_channel
        extend=1
        for v in cfg[:-1]:
            conv2d = nn.ConvTranspose2d( in_channels, v[0], kernel_size=2+v[1], stride=v[1], padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v[0]
            extend*=v[1]
        conv2d = nn.ConvTranspose2d( in_channels, cfg[-1][0], kernel_size=2+cfg[-1][1], stride=cfg[-1][1], padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(cfg[-1][0]), nn.Tanh()]
        else:
            layers += [conv2d, nn.Tanh()]
        extend*=cfg[-1][1]
        return nn.Sequential(*layers), extend
    def optimizer(self, lr=0.0001, betas= (0.5,0.999)):
        return torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, label_dim):
        super(Discriminator, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d( input_size, hidden_size, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(hidden_size * 2)
        self.conv3 = nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(hidden_size * 4)
        self.conv4 = nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(hidden_size * 8)
        self.conv5 = nn.Conv2d(hidden_size * 8, hidden_size * 1, 4, 1, 0, bias=False)
        self.disc_linear = nn.Linear(hidden_size * 1, 1)
        self.aux_linear = nn.Linear(hidden_size * 1, label_dim)
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.LeakyReLU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)

        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.LeakyReLU(x)

        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        c = self.aux_linear(x)
        c = self.softmax(c)
        s = self.disc_linear(x)
        s = self.sigmoid(s)
        return s,c
    def make_layers(self, input_channel, cfg,  batch_norm=False):
        #cfg = [(64,2), (64,2)]
        layers = []
        in_channels = input_channel
        compress=1
        for v in cfg[:-1]:
            conv2d = nn.Conv2d( in_channels, v[0], kernel_size=2+v[1], stride=v[1], padding=1,bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v[0]),nn.LeakyReLU(0.2)]
            else:
                layers += [conv2d, nn.LeakyReLU(0.2)]
            in_channels = v[0]
            compress*=v[1]
        conv2d = nn.Conv2d( in_channels, cfg[-1][0], kernel_size=2+cfg[-1][1], stride=cfg[-1][1], padding=1,bias=False)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(cfg[-1][0]),nn.Sigmoid()]
        else:
            layers += [conv2d, nn.Sigmoid()]
        compress*=cfg[-1][1]
        return nn.Sequential(*layers), compress
    def optimizer(self, lr=0.0001, betas= (0.5,0.999)):
        return torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
class FaceDataset(Dataset):
    def __init__(self, image, c, n_hair_c, n_eyes_c, flip=True, rotate= False):
        self.image = image
        self.c = c
        self.n_hair_c = n_hair_c
        self.n_eyes_c = n_eyes_c
        self.flip_n= int(flip)+1
        self.rotate= rotate
    def __getitem__(self, i):
        index= i// self.flip_n 
        flip = bool( i % self.flip_n )
        if flip == True: x= np.flip(self.image[index],2).copy()
        else: x= self.image[index]
        x=(torch.FloatTensor(x)- 127.5)/127.5
        if self.rotate: x=torchvision.transforms.RandomRotation(5)
        c=torch.FloatTensor([int(i==self.c[index,0]) for i in range(self.n_hair_c)] + [int(i==self.c[index,1]) for i in range(self.n_eyes_c)])
        return x, c
    def __len__(self):
        return len(self.image)*self.flip_n
class Color:
    def __init__(self, vocabulary_file= None):
        if vocabulary_file == None:
            self.color2index= {}
            self.color2count = {}
            self.index2color = {}
            self.n_colors = 0  # Count SOS and EOS and PAD and UNK
        else:
            self.load(vocabulary_file)
    def addColors(self, sentences):
        sen= sentences.split(' ')
        return self.addColor(sen[0]), self.addColor(sen[2])
    def addColor(self, color):
        color=color.lower()
        if color in self.color2count: self.color2count[color]+=1
        else:
            self.color2index[color] = self.n_colors
            self.color2count[color] = 1
            self.index2color[self.n_colors] = color
            self.n_colors += 1
        return self.color2index[color]
    def save(self, path):
        index_list= sorted( self.color2index, key= self.color2index.get)
        with open( path, 'w') as f:
            f.write('\n'.join(index_list))
    def load(self, path):
        self.color2index= {}
        self.color2count= {}
        self.index2color= {}
        with open(path,'r') as f:
            i=0
            for line in f:
                color=line.replace('\n','')
                self.color2index[color] = i
                self.color2count[color]=0
                self.index2color[i] = color
                i+=1
            self.n_words=len(self.color2index)
