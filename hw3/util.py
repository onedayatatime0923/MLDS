
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
#from tensorboardX import SummaryWriter 
import matplotlib.pyplot as plt
assert torch and nn and Variable and F and Dataset and DataLoader
assert time and np


class DataManager():
    def __init__(self,latent_dim=0, discriminator_update_num=0, generator_update_num=0):
        self.data={}
        self.discriminator_update_num= discriminator_update_num
        self.generator_update_num= generator_update_num
        self.latent_dim= latent_dim
        self.data_size= 0
        self.label_dim= 4
        self.writer= None
    def tb_setting(self, path):
        for f in os.listdir(path): 
            os.remove('{}/{}'.format(path,f))
        #self.writer = SummaryWriter(path)
    def tb_graph(self, model, input_shape):
        if isinstance(input_shape, tuple):
            dummy_input= Variable( torch.rand(1, *input_shape).cuda())
        elif isinstance(input_shape, int):
            dummy_input= Variable( torch.rand(1, input_shape).cuda())
        else: raise ValueError('Wrong input_shape')
        self.writer.add_graph(nn.Sequential(*model), (dummy_input, ))
    def get_data(self,name, i_path, c_path= None, class_range=None, mode= 'gan',batch_size= 128, shuffle=False):
        x=[]
        for p in i_path:
            file_list = [file for file in os.listdir(p) if file.endswith('.png')]
            file_list.sort()
            for i in file_list:
                x.append(np.array(Image.open('{}/{}'.format(p,i)),dtype=np.uint8).transpose((2,0,1)))
                print('\rreading {} image...{}'.format(name,len(x)),end='')

        x=np.array(x)
        self.data_size= x.shape[1:]
        print('\rreading {} image...finish'.format(name))

        if class_range!= None:
            y=[]
            for p in c_path:
                with open(p, 'r') as f:
                    next(f)
                    for line in f:
                        data=[int(i=='1.0') for i in line.split(',')[1:]]
                        y.append(np.array(data,dtype=np.uint8))
                    print('\rreading {} class...{}'.format(name,len(y)),end='')
            y=np.array(y)[:,class_range[0]:class_range[1]]
            self.label_dim= y.shape[1]
            print('\rreading {} class...finish'.format(name))
            self.data[name]=DataLoader(ImageDataset(x, y ,mode, flip= False),batch_size=batch_size, shuffle=shuffle)
            return x.shape[1:], y.shape[1]
        else:
            self.data[name]=DataLoader(ImageDataset(x, None ,mode, flip= False),batch_size=batch_size, shuffle=shuffle)
            return x.shape[1:]
    def train_acgan(self,name, generator, discriminator, optimizer, epoch, print_every=1):
        start= time.time()
        generator.train()
        discriminator.train()
        
        generator_optimizer= optimizer[0]
        discriminator_optimizer= optimizer[1]
        
        criterion= nn.BCELoss()
        total_loss= [0,0,0]     # G, D, C
        batch_loss= [0,0,0]     # G, D, C
        
        data_size= len(self.data[name].dataset)
        for j, (i, c) in enumerate(self.data[name]):
            batch_index=j+1
            origin_i = Variable(i).cuda()
            origin_c = Variable(c).cuda()
            # update discriminator
            for k in range(self.discriminator_update_num):
                latent_c = torch.zeros(len(i),self.label_dim).random_(0,2)
                latent = Variable(torch.cat((torch.randn(len(i),self.latent_dim),latent_c),1).cuda())
                fake_i, fake_c= discriminator(generator(latent))
                real_i, real_c= discriminator(origin_i)
                zero= Variable( torch.rand(len(i),1)*0.3).cuda()
                one= Variable( torch.rand(len(i),1)*0.5 + 0.7).cuda()
                loss_fake_i= criterion( fake_i, zero)
                loss_real_i= criterion( real_i, one)
                loss_fake_c= criterion( fake_c, Variable(latent_c.cuda()))
                loss_real_c= criterion( real_c, origin_c)
                loss= (loss_fake_i + loss_fake_c + loss_real_i + loss_real_c) /4
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()
                batch_loss[0]+= float(loss)
                #print(float(loss))

            # update generator
            for k in range(self.generator_update_num):
                latent_c = torch.zeros(len(i),self.label_dim).random_(0,2)
                latent = Variable(torch.cat((torch.randn(len(i),self.latent_dim),latent_c),1).cuda())
                fake_i, fake_c= discriminator(generator(latent))
                one= Variable( torch.rand(len(i),1)*0.5 + 0.7).cuda()
                loss_fake_i= criterion( fake_i, one)
                loss_fake_c= criterion( fake_c, Variable(latent_c.cuda()))
                loss= (loss_fake_i + loss_fake_c ) /2
                #loss=  torch.mean(-torch.log(discriminator(generator(batch_x))))
                generator_optimizer.zero_grad()
                loss.backward()
                generator_optimizer.step()
                batch_loss[1]+= float(loss)
                #print(float(loss))

            if batch_index% print_every == 0:
                total_loss[0]+= batch_loss[0]/ (self.discriminator_update_num ) if (self.discriminator_update_num!=0) else 0
                total_loss[1]+= batch_loss[1]/ (self.generator_update_num ) if (self.generator_update_num!=0) else 0
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | G Loss: {:.6f} | D Loss: {:.6f} | Time: {}  '.format(
                                epoch , batch_index*len(i), data_size, 
                                100. * batch_index*len(i)/ data_size,
                                batch_loss[1]/ (self.generator_update_num *print_every) if (self.generator_update_num!=0) else 0,
                                batch_loss[0]/ (self.discriminator_update_num *print_every)if (self.discriminator_update_num!=0) else 0,
                                self.timeSince(start, batch_index*len(i)/ data_size)),end='')
                batch_loss= [0,0]
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Total G Loss: {:.6f} | Total D Loss: {:.6f} | Time: {}  '.format(
                        epoch , data_size, data_size, 100. ,
                        float(total_loss[1])/batch_index,float(total_loss[0])/batch_index,
                        self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('discriminator loss', float(total_loss[0])/ batch_index, epoch)
            self.writer.add_scalar('generator loss', float(total_loss[1])/ batch_index, epoch)
        print('-'*80)
        return total_loss[0]/ batch_index, total_loss[1]/ batch_index
    def val_acgan(self, generator, discriminator, epoch=None, n=20, path=None, sample_i_path= None):
        generator.eval()
        discriminator.eval()
        
        x= torch.randn(n,self.latent_dim)
        predict= []
        c_no= torch.FloatTensor([[ 0 for i in range(self.label_dim)]]).repeat(n,1)
        latent_no= Variable(torch.cat((x, c_no),1).cuda())
        predict.extend(generator(latent_no).cpu().data.unsqueeze(1))
        #for l in range(self.label_dim):
        l=1
        c_yes= torch.FloatTensor([[ int( i == l) for i in range(self.label_dim)]]).repeat(n,1)
        latent_yes= Variable(torch.cat((x, c_yes),1).cuda())
        predict.extend(generator(latent_yes).cpu().data.unsqueeze(1))

        predict= torch.cat(predict,0)
        if self.writer!=None:
            self.write(predict,path,'gan')
            self.writer.add_image('sample image result', torchvision.utils.make_grid(predict, nrow=n, normalize=True, range=(-1,1)), epoch)
        if sample_i_path != None: self.plot_grid(torchvision.utils.make_grid((predict*127.5)+127.5, nrow= n), sample_i_path)
    def train_gan(self,name, generator, discriminator, optimizer, epoch, print_every=1):
        start= time.time()
        generator.train()
        discriminator.train()
        
        generator_optimizer= optimizer[0]
        discriminator_optimizer= optimizer[1]
        
        criterion= torch.nn.BCELoss()
        total_loss= [0,0]
        batch_loss= [0,0]
        
        data_size= len(self.data[name].dataset)
        for i, y in enumerate(self.data[name]):
            batch_index=i+1
            batch_y = Variable(y).cuda()
            # update discriminator
            for j in range(self.discriminator_update_num):
                batch_x = Variable(torch.randn(len(y),self.latent_dim).cuda())
                #loss_gen= torch.mean( -torch.log(1-discriminator(generator(batch_x))))
                #loss_dis= torch.mean( -torch.log(discriminator(batch_y)))
                zero= Variable( torch.rand(len(y),1)*0.3).cuda()
                one= Variable( torch.rand(len(y),1)*0.5 + 0.7).cuda()
                loss_fake= criterion(discriminator(generator(batch_x)), zero)
                loss_real= criterion(discriminator(batch_y), one)
                loss= (loss_fake + loss_real) 
                '''
                if epoch== 3:
                    self.write(generator(batch_x[: 3]).cpu().data,'./data/gan','gan')
                    print('fake')
                    input()
                    self.write(batch_y[: 3].cpu().data,'./data/gan','gan')
                    print('real')
                    input()
                    print(discriminator(batch_Dx[:10]))
                    print(batch_Dy[:10])
                    print(discriminator(batch_Dx[-10:]))
                    print(batch_Dy[-10:])
                    print(discriminator(batch_Dx).size())
                    print(float(loss))
                    input()
                '''
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()
                batch_loss[0]+= float(loss)
                #print(float(loss))

            # update generator
            for j in range(self.generator_update_num):
                batch_x = Variable(torch.randn(len(y),self.latent_dim).cuda())
                one= Variable( torch.rand(len(y),1)*0.5 + 0.7).cuda()
                loss= criterion(discriminator(generator(batch_x)), one)
                '''
                if epoch== 3:
                    print(discriminator(batch_Dx[:10]))
                    print(batch_Dy[:10])
                    print(discriminator(batch_Dx[-10:]))
                    print(batch_Dy[-10:])
                    print(discriminator(batch_Dx).size())
                    print(float(loss))
                    input()
                '''
                #loss=  torch.mean(-torch.log(discriminator(generator(batch_x))))
                generator_optimizer.zero_grad()
                loss.backward()
                generator_optimizer.step()
                batch_loss[1]+= float(loss)
                #print(float(loss))

            if batch_index% print_every == 0:
                total_loss[0]+= batch_loss[0]/ (self.discriminator_update_num ) if (self.discriminator_update_num!=0) else 0
                total_loss[1]+= batch_loss[1]/ (self.generator_update_num ) if (self.generator_update_num!=0) else 0
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | G Loss: {:.6f} | D Loss: {:.6f} | Time: {}  '.format(
                                epoch , batch_index*len(batch_x), data_size, 
                                100. * batch_index*len(batch_x)/ data_size,
                                batch_loss[1]/ (self.generator_update_num *print_every) if (self.generator_update_num!=0) else 0,
                                batch_loss[0]/ (self.discriminator_update_num *print_every)if (self.discriminator_update_num!=0) else 0,
                                self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')
                batch_loss= [0,0]
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Total G Loss: {:.6f} | Total D Loss: {:.6f} | Time: {}  '.format(
                        epoch , data_size, data_size, 100. ,
                        float(total_loss[1])/batch_index,float(total_loss[0])/batch_index,
                        self.timeSince(start, 1)))
        if self.writer!=None:
            self.writer.add_scalar('discriminator loss', float(total_loss[0])/ batch_index, epoch)
            self.writer.add_scalar('generator loss', float(total_loss[1])/ batch_index, epoch)
        print('-'*80)
        return total_loss[0]/ batch_index, total_loss[1]/ batch_index
    def val_gan(self, generator, discriminator, epoch=0, n=20, path=None, sample_i_path = None):
        generator.eval()
        discriminator.eval()
        
        batch_x = Variable(torch.randn(n,self.latent_dim).cuda())
        predict= generator(batch_x).cpu().data
        #self.write(predict,path,'gan')

        if self.writer!=None:
            self.writer.add_image('sample image result', torchvision.utils.make_grid(predict, normalize=True, range=(-1,1)), epoch)
        if sample_i_path != None: self.plot_grid(torchvision.utils.make_grid((predict*127.5)+127.5), sample_i_path)
    def train_vae(self,name, encoder, decoder, optimizer, epoch, kl_coefficient=5E-5, print_every=5):
        start= time.time()
        encoder.train()
        decoder.train()
        
        encoder_optimizer= optimizer[0]
        decoder_optimizer= optimizer[1]
        
        criterion = nn.MSELoss()
        total_loss=torch.zeros(2)
        batch_loss=torch.zeros(2)
        
        data_size= len(self.data[name].dataset)
        for i, (x, c) in enumerate(self.data[name]):
            batch_index=i+1
            batch_x= Variable(x).cuda()
            mean_x, logvar_x= encoder(batch_x)
            output= decoder(mean_x, logvar_x,mode='train')
            #output = output.view(-1,output.size()[2])
            reconstruction_loss = criterion(output,batch_x)
            kl_divergence_loss= torch.sum((-1/2)*( 1+ logvar_x- mean_x**2 - torch.exp(logvar_x)))/(len(x))
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            (reconstruction_loss+ kl_coefficient*kl_divergence_loss).backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            batch_loss+= torch.FloatTensor([float(reconstruction_loss), float(kl_divergence_loss)]) # sum up batch loss
            total_loss+= torch.FloatTensor([float(reconstruction_loss), float(kl_divergence_loss)]) # sum up total loss
            if batch_index% print_every == 0:
                print_loss= batch_loss / print_every
                batch_loss=torch.zeros(2)
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | MSE Loss: {:.6f} | KL Loss: {:.6f} | Time: {}  '.format(
                                epoch , batch_index*len(batch_x), data_size, 100. * batch_index*len(batch_x)/ data_size, 
                                float(print_loss[0]),float(print_loss[1]),
                                self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Total MSE Loss: {:.6f} | Total KL Loss: {:.6f} | Time: {}  '.format(
                        epoch , data_size, data_size, 100,
                        float(total_loss[0]/ batch_index), float(total_loss[1]/ batch_index),
                        self.timeSince(start, 1)))
        print('-'*80)
        self.writer.add_scalar('Train Reconstruction loss', float(total_loss[0])/ batch_index, epoch)
        self.writer.add_scalar('Train KL Divergance loss', float(total_loss[1])/ batch_index, epoch)
        return total_loss[0]/ batch_index, total_loss[1]/ batch_index
    def val_vae(self,name, encoder, decoder, optimizer, epoch=0, print_every=5, reconstruct_n = 10, sample_n= 32,  path=None,reconstruct_i_path=None,sample_i_path=None):
        start= time.time()
        encoder.eval()
        decoder.eval()

        criterion = nn.MSELoss()
        total_loss=torch.zeros(2)
        ground =[]
        predict=[]
        
        data_size= len(self.data[name].dataset)
        for i, (x, c) in enumerate(self.data[name]):
            batch_index=i+1
            batch_x= Variable(x).cuda()
            mean_x, logvar_x= encoder(batch_x)
            output= decoder(mean_x, logvar_x, mode='test')
            reconstruction_loss = criterion(output,batch_x)
            kl_divergence_loss= torch.sum((-1/2)*( 1+ logvar_x- mean_x**2 - torch.exp(logvar_x)))/(len(x))
            # loss
            total_loss+= torch.FloatTensor([float(reconstruction_loss), float(kl_divergence_loss)]) # sum up total loss
            # ground
            ground.extend(x.unsqueeze(0))
            # predict
            predict.extend(output.cpu().data.unsqueeze(0))
            if batch_index % print_every == 0:
                print('\rVal: [{}/{} ({:.0f}%)] | Time: {}'.format(
                         batch_index * len(x), data_size,
                        100. * batch_index* len(x) / data_size,
                        self.timeSince(start, batch_index*len(x)/ data_size)),end='')

        total_loss/=  batch_index
        ground=torch.cat(ground,0)
        predict=torch.cat(predict,0)

        sample_x = Variable(torch.normal(torch.zeros(sample_n,self.latent_dim))).cuda()
        predict= torch.cat((decoder(sample_x,mode='test').cpu().data,predict),0)
        print('\nVal set: Average Reconstruction Loss: {:.6f} | Average KL Divergance Loss: {:.6f} | Time: {}  '.format(float(total_loss[0]),float(total_loss[1]),
                        self.timeSince(start,1)))
        print('-'*80)
        #self.write(predict,path,'vae')
        if self.writer!=None:
            self.writer.add_scalar('Test Reconstruction loss', float(total_loss[0]), epoch)
            self.writer.add_scalar('Test KL Divergance loss', float(total_loss[1]), epoch)
            self.writer.add_image('sample image result', torchvision.utils.make_grid(predict[:sample_n], normalize=True, range=(0,1)), epoch)
        reconstruct=torch.cat((ground[:reconstruct_n], predict[sample_n:sample_n+reconstruct_n]),0)
        if reconstruct_i_path != None: self.plot_grid(torchvision.utils.make_grid(reconstruct*255,nrow=reconstruct_n), reconstruct_i_path)
        if sample_i_path != None: self.plot_grid(torchvision.utils.make_grid(predict[:sample_n]*255), sample_i_path)
        return total_loss[0], total_loss[1]
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
            if mode== 'vae':
                im=data[i].transpose((1,2,0))*255
            elif mode== 'gan':
                im=(data[i].transpose((1,2,0))*127.5)+127.5
            else: raise ValueError('Wrong mode')
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
    def __init__(self, input_size, hidden_size ):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (input_size) x 64 x 64
            nn.Conv2d(input_size, hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size) x 32 x 32
            nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*2) x 16 x 16
            nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*4) x 8 x 8
            nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*8) x 4 x 4
            nn.Conv2d(hidden_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0),-1)
        return x
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
class Discriminator_Acgan(nn.Module):
    def __init__(self, input_size, hidden_size, label_dim):
        super(Discriminator_Acgan, self).__init__()
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
class ImageDataset(Dataset):
    def __init__(self, image, c, mode, flip=True, rotate= False):
        self.image = image
        self.c = c
        self.mode = mode
        self.flip_n= int(flip)+1
        self.rotate= rotate
    def __getitem__(self, i):
        index= i// self.flip_n 
        flip = bool( i % self.flip_n )

        if self.mode=='vae':
            if flip == True: x= np.flip(self.image[index],2).copy()
            else: x= self.image[index]
            x=torch.FloatTensor(x)/255
            if self.rotate: x=torchvision.transforms.RandomRotation(5)
            if not isinstance(self.c, np.ndarray) : return x
            c=torch.FloatTensor(self.c[index][:])
            return x, c
        elif self.mode=='gan':
            if flip == True: x= np.flip(self.image[index],2).copy()
            else: x= self.image[index]
            x=(torch.FloatTensor(x)- 127.5)/127.5
            if self.rotate>0: x=torchvision.transforms.RandomRotation(5)
            if not isinstance(self.c, np.ndarray) : return x
            c=torch.FloatTensor(self.c[index][:])
            return x, c
        else: raise ValueError('Wrong mode.')
    def __len__(self):
        return len(self.image)*self.flip_n
