
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch import optim
import os
import numpy as np
import json
import random
import time
import math
assert os and np and F

use_cuda=True
MAX_LENGTH=0

class Datamanager:
    def __init__(self):
        self.voc=Vocabulary()
        self.data={}
        self.vocab_size=0
        self.data_size=0
        self.max_length=0
    def get_data(self,name,f_path,l_path,batch_size):
        feats={}
        captions={}
        data_size=0
        max_sen=0
        for i in os.listdir(f_path):
            if not i.startswith('.'):
                x=torch.FloatTensor(np.load('{}/{}'.format(f_path,i)))
                feats[i[:-4]]=x

        with open(l_path) as f:
            labels=json.load(f)
        for l in labels:
            data_size+=len(l['caption'])
            m=self.voc.addSentence(l['caption'])
            if m>max_sen: max_sen=m
        self.data_size=data_size
        self.max_length=max_sen+2
        self.vocab_size=self.voc.n_words
        for l in labels:
            captions[l['id']]=self.IndexFromSentence(l['caption'],begin=True,end=True)

        dataset=VideoDataset(feats,captions)
        self.data[name]=DataLoader(dataset, batch_size=batch_size, shuffle=True)
    def generate_data(self,name,batch_size,shuffle=True):
        [feats,labels]=self.data[name]
        indexes=[(i,j) for i in range(len(labels)) for j in range(len(labels[i]['caption']))]
        if shuffle: random.shuffle(indexes)
        for i in range(0,self.data_size-batch_size, batch_size):
            x,y=[],[]
            for  j in range(i, i+batch_size):
                x.append(feats[labels[indexes[j][0]]['id']+'.npy'])
                y.append(self.IndexFromSentence(labels[indexes[j][0]]['caption'][indexes[j][1]],begin=True,end=True))
            x=Variable(torch.FloatTensor(np.array(x))).cuda()
            y=Variable(torch.LongTensor(np.array(y))).cuda()
            yield x,y
        x,y=[],[]
        for  j in range(self.data_size// batch_size *batch_size, self.data_size):
            x.append(feats[labels[indexes[j][0]]['id']+'.npy'])
            y.append(self.IndexFromSentence(labels[indexes[j][0]]['caption'][indexes[j][1]],begin=True,end=True))
        x=Variable(torch.FloatTensor(np.array(x))).cuda()
        y=Variable(torch.LongTensor(np.array(y))).cuda()
        yield x,y
    def IndexFromSentence(self,sentences,begin=False,end=True):
        indexes=[]
        for s in sentences:
            index=[]
            if begin: index.append(self.voc.word2index['SOS'])
            index.extend([self.voc.word2index[word] for word in s.split(' ')])
            if end: index.append(self.voc.word2index['EOS'])
            if len(index)< self.max_length : 
                index.extend([self.voc.word2index['PAD'] for i in range(self.max_length  -len(index))])
            indexes.append(index)
        indexes = torch.LongTensor(indexes)
        return indexes
    def train(self,input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=1):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = 0

        encoder_outputs, encoder_hidden = encoder(input_variable)


        decoder_hidden= encoder_hidden
        decoder_input=torch.index_select(target_variable, 1, Variable(torch.LongTensor([0])).cuda())
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(1,self.max_length):
                decoder_output, decoder_hidden= decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_input=torch.index_select(target_variable, 1, Variable(torch.LongTensor([di])).cuda())
                target=decoder_input.view(-1)
                loss += criterion(decoder_output, target)

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(1,self.max_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                ni = decoder_output.data.max(1,keepdim=True)[1]
                decoder_input = Variable(torch.LongTensor(ni)).cuda()
                target=torch.index_select(target_variable, 1, Variable(torch.LongTensor([di])).cuda()).view(-1)
                loss += criterion(decoder_output, target_variable[:di])

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.data[0]/ self.max_length
    def trainIters(self,encoder, decoder, name, n_epochs, learning_rate=0.001, print_every=2, plot_every=100):
        plot_losses = []

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, n_epochs+ 1):
            start = time.time()
            for step, (batch_x, batch_y) in enumerate(self.data[name]):
                batch_index=step+1
                loss_total=0
                print_loss_total = 0  # Reset every print_every
                plot_loss_total = 0  # Reset every plot_every
                batch_x=Variable(batch_x).cuda()
                batch_y=Variable(batch_y).cuda()

                loss = self.train(batch_x, batch_y, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
                loss_total+=loss

                if step % print_every == 0:
                    print_loss_avg = (loss_total - print_loss_total )/ print_every
                    print_loss_total = loss_total
                    print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] |  Loss: {:.6f} | Time: {}  '.format(
                                epoch , batch_index*len(batch_x), self.data_size,
                                100. * batch_index*len(batch_x)/ self.data_size, print_loss_avg,
                                self.timeSince(start, batch_index*len(batch_x)/ self.data_size)),end='')
                if epoch % plot_every == 0:
                    plot_loss_avg = (loss_total - plot_loss_total )/ plot_every
                    print_loss_total = loss_total
                    plot_losses.append(plot_loss_avg)
            print('\nTime: {} | Total loss: {:.4f}'.format(self.timeSince(start,1),loss_total/batch_index))
            print('-'*60)
        return plot_losses
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
class Vocabulary:
    def __init__(self):
        self.word2index = {"SOS":0, "EOS":1, "PAD":2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3  # Count SOS and EOS
    def addSentence(self, sentences):
        max_sen=0
        for sentence in sentences:
            sentence_list=sentence.split(' ')
            for word in sentence_list:
                self.addWord(word)
            if len(sentence_list)>max_sen: max_sen=len(sentence_list)
        return max_sen
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
class EncoderRNN(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.hidden= self.initHidden()
        self.gru = nn.GRU(input_size, hidden_size,batch_first=True)
    def forward(self, x):
        hidden= torch.cat([self.hidden for i in range(len(x))],1)
        output, hidden = self.gru(x, hidden)
        return output, hidden
    def initHidden(self):
        return Variable(torch.zeros(1,1, self.hidden_size)).cuda()
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout_p=0.1 ):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn_weight = nn.Softmax(1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size,batch_first=True)
        self.out = nn.Sequential( nn.Linear(self.hidden_size, self.vocab_size))
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x).squeeze(1)

        z = self.attn(torch.cat((embedded, hidden.squeeze(0)), 1)).unsqueeze(2)
        attn_weights = self.attn_weight(torch.bmm(encoder_outputs,z).squeeze(2)).unsqueeze(1)
        attn_applied = torch.bmm(attn_weights,encoder_outputs).squeeze(1)

        output = self.attn_combine(torch.cat((embedded, attn_applied), 1).unsqueeze(1))

        output, hidden = self.gru(output, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden 
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result.cuda()
class VideoDataset(Dataset):
    def __init__(self, feats, captions):
        self.feats=feats
        self.captions=captions
        index=[]
        for i in captions:
            index.extend([(i,j) for j in range(len(captions[i]))])
        self.index=index
    def __getitem__(self, i):
        x=self.feats[self.index[i][0]]
        y=self.captions[self.index[i][0]][1]
        return x,y
    def __len__(self):
        return len(self.index)
