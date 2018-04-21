
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
assert os and np and F and math


class Datamanager:
    def __init__(self):
        self.voc=Vocabulary()
        self.data={}
        self.vocab_size=0
        self.data_size=0
        self.test_data_size=0
        self.test_data_labels={}
        self.max_length=0
        self.print_image=None
    def get_data(self,name,c_path,mode,batch_size,shuffle=True):
        feats = {}
        target = {}
        split_corpus = []
        with open(c_path,'r',encoding='utf-8') as f :
            tmp = []
            while(True):
                data = f.readline().replace('\n','')
                #print(data)
                if(data==''): break
                elif(data=='+++$+++'): 
                    split_corpus.append(tmp)
                    tmp = []
                    if(len(split_corpus)==1000): break
                    #print(tmp)
                    continue
                else: tmp.append(data)
            #print(len(split_corpus))
            #print(split_corpus[-1])  
            #input()
        f.close()
        print('finish reading')

        self.vocab_size=self.voc.n_words
        max_sen = 0

        for dialog in split_corpus:
            for sen in dialog:
                if(sen=='+++$+++'): continue
                m = self.voc.addSentence(sen)
                #print(sen,' ',m)
                if m > max_sen: max_sen = m
                #print(max_sen)
        self.max_length = max_sen + 2
        #print(self.max_length)
        
        count_f = 0
        count_t = 0
        count=0
        for dialog in split_corpus:
            if(len(dialog)<2): continue
            print('count= ',count)
            count+=1
            for num,sen in enumerate(dialog):
                if(num==0):
                    #print('A')
                    feats[count_f] = self.IndexFromSentence(sen,begin=True,end=True)
                    #feats[count_f] = sen
                    #print('feats {}: {}'.format(count_f,sen))
                    count_f+=1
                elif(num==len(dialog)-1):
                    #print('B')
                    target[count_t] = self.IndexFromSentence(sen,begin=True,end=True)
                    #target[count_t] = sen
                    #print('target {}: {}'.format(count_t,sen))
                    count_t+=1
                else:
                    #print('C')
                    feats[count_f] = self.IndexFromSentence(sen,begin=True,end=True)
                    target[count_t] = self.IndexFromSentence(sen,begin=True,end=True)
                    #feats[count_f] = sen
                    #target[count_t] = sen
                    #print('feats {}: {}'.format(count_f,sen))
                    #print('target {}: {}'.format(count_t,sen))
                    count_f+=1              
                    count_t+=1
            #print(count_f,' ',count_t)
        #input()
        print('target length: ',len(target))
        print('feats length: ',len(feats))
        self.data_size = len(feats)
        #print(feats[0].size())
        #input()
        dataset=DialogDataset(feats,target) # feat shape torch.Size([28])
        self.data[name]=DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    def IndexFromSentence(self,s,begin=False,end=True):
        #indexes=[]
        #for s in sentence:
        index=[]
        if begin: index.append(self.voc.word2index('SOS'))
        index.extend([self.voc.word2index(word) for word in s.split(' ')])
        if end: index.append(self.voc.word2index('EOS'))
        if len(index)< self.max_length : 
            index.extend([self.voc.word2index('PAD') for i in range(self.max_length - len(index))])
        #indexes.append(index)
        index = torch.LongTensor(index)
        return index
    def train(self,input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=1):
        encoder.train()
        decoder.train()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = 0
        print('start encoding ...')
        encoder_outputs, encoder_hidden = encoder(input_variable)
        print('finish encoding ...')

        decoder_hidden= decoder.hidden_layer(len(input_variable))
        print('*'*50)
        target_variable = target_variable.view(len(target_variable),self.max_length)
        print('target_variable size= ',target_variable.size())
        
        decoder_input = torch.index_select(target_variable, 1 , Variable(torch.LongTensor([0])).cuda())
        print('decoder_input size= ',decoder_input.size())
        print('start decoding ...')
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(1,self.max_length):
                decoder_output, decoder_hidden= decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_input=torch.index_select(target_variable, 1, Variable(torch.LongTensor([di])).cuda())
                target=decoder_input.view(-1)
                loss += self.loss(criterion,decoder_output, target)

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(1,self.max_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                ni = decoder_output.data.max(1,keepdim=True)[1]
                decoder_input = Variable(ni)
                target=torch.index_select(target_variable, 1, Variable(torch.LongTensor([di])).cuda()).view(-1)
                loss += self.loss(criterion,decoder_output, target)
        print('finish decoding ...')
        print('updating parameters ...')
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.data[0]/ (self.max_length)
        
    def trainIters(self,encoder, decoder, name, test_name, n_epochs, learning_rate=0.001, print_every=2, plot_every=100):
        plot_losses = []

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        
        criterion = nn.CrossEntropyLoss(size_average=False)
        for epoch in range(1, n_epochs+ 1):
            start = time.time()
            loss_total=0
            print_loss_total = 0  # Reset every print_every
            plot_loss_total = 0  # Reset every plot_every
            for step, (batch_x, batch_y) in enumerate(self.data[name]):
                batch_index=step+1
                batch_x=Variable(batch_x).cuda()
                batch_y=Variable(batch_y).cuda()

                loss = self.train(batch_x, batch_y, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
                loss_total+=loss

                if batch_index% print_every == 0:
                    print_loss_avg = (loss_total - print_loss_total )/ print_every
                    print_loss_total = loss_total
                    print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] |  Loss: {:.6f} | Time: {}  '.format(
                                epoch , batch_index*len(batch_x), self.data_size,
                                100. * batch_index*len(batch_x)/ self.data_size, print_loss_avg,
                                self.timeSince(start, batch_index*len(batch_x)/ self.data_size)),end='')
                if batch_index% plot_every == 0:
                    plot_loss_avg = (loss_total - plot_loss_total )/ plot_every
                    print_loss_total = loss_total
                    plot_losses.append(plot_loss_avg)
            print('\nTime: {} | Total loss: {:.4f}'.format(self.timeSince(start,1),loss_total/batch_index))
            print('-'*60)
            self.evaluate(encoder,decoder,test_name)
        return plot_losses
    def evaluate(self,encoder, decoder, name,print_image=None):
        encoder.eval()
        decoder.eval()

        start = time.time()
        loss=0
        decoded_words = []
        criterion = nn.CrossEntropyLoss(size_average=False)

        if print_image==None and self.print_image== None:
            self.print_image=[random.choice(list(self.data[name].dataset.feats.keys())) for i in  range(5)]
        elif print_image!=None :
            self.print_image=print_image

        for step, (batch_x, batch_y) in enumerate(self.data[name]):
            batch_index = step + 1
            batch_x=Variable(batch_x).cuda()
            batch_y=Variable(batch_y).cuda()

            encoder_outputs, encoder_hidden = encoder(batch_x)

            decoder_hidden= encoder_hidden 
            decoder_input=torch.index_select(batch_y, 1, Variable(torch.LongTensor([0])).cuda())

            words=[]

            for di in range(1,self.max_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                ni = decoder_output.data.max(1,keepdim=True)[1]
                decoder_input = Variable(ni)       
                words.append(ni)

                target=torch.index_select(batch_y, 1, Variable(torch.LongTensor([di])).cuda()).view(-1)
                loss += float(self.loss(criterion,decoder_output, target))

            words= torch.cat(words,1).unsqueeze(1)
            decoded_words.extend(words)
            loss /= self.max_length

            print('\rTest | [{}/{} ({:.0f}%)] |  Loss: {:.6f} | Time: {}  '.format(
                        batch_index*len(batch_x), self.test_data_size,
                        100. * batch_index*len(batch_x)/ self.test_data_size, loss,
                        self.timeSince(start, batch_index*len(batch_x)/ self.data_size)),end='')

        print('\nTime: {} | Total loss: {:.4f}'.format(self.timeSince(start,1),loss))
        decoded_words=torch.cat(decoded_words,0)
        for i in self.print_image:
            #seq_id=self.data[name].dataset.get_id(i)
            seq_list=[]
            for j in decoded_words[i]:
                if j == self.voc.word2index('EOS'): break
                seq_list.append(self.voc.index2word[j])
            d_seq = ' '.join(seq_list)
            g_seq = self.data[name].dataset.target[i]
            print('decoded_sequence: {}'.format(d_seq))
            print('ground_sequence: {}'.format(g_seq))
        print('-'*60)

        return decoded_words
    def loss(self,criterion,output,target):
        check_t=(target!=self.voc.word2index("PAD"))
        t=torch.masked_select(target,check_t).view(-1)
        check_o=check_t.view(-1,1)
        o=torch.masked_select(output,check_o).view(-1,self.vocab_size)
        if len(t)==0: return 0
        else : return criterion(o,t)/len(t)
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
        self.w2i= {"SOS":0, "EOS":1, "PAD":2, "UNK":3}
        self.min_count = 1
        self.word2count = {"SOS":self.min_count, "EOS":self.min_count, "PAD":self.min_count, "UNK":self.min_count}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD", 3:"UNK"}
        self.n_words = 4  # Count SOS and EOS and PAD and UNK
    def word2index(self,word):
        if word in self.w2i and self.word2count[word] >= self.min_count: return self.w2i[word]
        else: return self.w2i["UNK"]
    def addSentence(self, sentence):
        sentence_list = sentence.split(' ')
        for word in sentence_list:
            self.addWord(word)
        max_sen = len(sentence_list)
        return max_sen
    def addWord(self, word):
        if word not in self.w2i:
            self.w2i[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        
class EncoderRNN(nn.Module):
    def __init__(self,input_size, hidden_size , vocab_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size,self.hidden_size)
        self.hidden= self.initHidden()
        self.gru = nn.GRU(input_size, hidden_size,batch_first=True)
    def forward(self, x):
        #x = x.view(-1,28) 
        embedded = self.embedding(x) # (batch_size,28,hidden_size)
        print('embedded size = ',embedded.size())
        
        hidden= torch.cat([self.hidden for i in range(len(x))],1) # (1,batch_size,hidden_size)
        print('hidden.size = ',hidden.size())
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
    def initHidden(self):
        return Variable(torch.zeros(1,1, self.hidden_size),requires_grad=True).cuda()
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.hidden= self.initHidden()

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn_weight = nn.Softmax(1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size,batch_first=True)
        self.out = nn.Sequential( nn.Linear(self.hidden_size, self.vocab_size))
    def forward(self, x, hidden, encoder_outputs):
        # x size: batch * 1
        # encoder outputs size: batch * 80 * hidden
        # hidden size: 1 * batch * hidden
        embedded = self.embedding(x).squeeze(1)         # batch *  hidden

        z = self.attn(torch.cat((embedded, hidden.squeeze(0)), 1)).unsqueeze(2)# batch * hidden * 1
        attn_weights = self.attn_weight(torch.bmm(encoder_outputs,z).squeeze(2)).unsqueeze(1)# batch * hidden * 1
        attn_applied = torch.bmm(attn_weights,encoder_outputs).squeeze(1)

        output = self.attn_combine(torch.cat((embedded, attn_applied), 1).unsqueeze(1))

        output, hidden = self.gru(output, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden 
    def hidden_layer(self,n):
        return  torch.cat([self.hidden for i in range(n)],1)
    def initHidden(self):
        return Variable(torch.zeros(1,1, self.hidden_size),requires_grad=True).cuda()
class DialogDataset(Dataset):
    def __init__(self, feats, target):
        self.feats = feats
        self.target = target
        #index=[]
        #id_={}
        #c=0
        #for i in captions:
            #id_[i]=c
            #c+=1
            #index.extend([(i,j) for j in range(len(captions[i]))])
        #self.id_= id_
        #self.index=index
        #print(len(index))
    #def get_id(self,name):
        #return self.id_[name]
    def __getitem__(self, i):
        #x=self.feats[self.index[i][0]]
        #y=self.captions[self.index[i][0]][self.index[i][1]]
        x = self.feats[i]
        y = self.target[i]
        return x,y
    def __len__(self):
        return len(self.feats)

