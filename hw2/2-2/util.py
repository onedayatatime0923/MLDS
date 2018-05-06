
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
assert os and np and F and math and json
 

class Datamanager:
    def __init__(self,min_count):
        self.voc=Vocabulary(min_count)
        self.data={}
        self.vocab_size=0
        self.data_size=0
        self.test_data_size=0
        self.test_data_labels={}
        self.max_length=0
        self.print_image=None
    def get_data(self,name,path,batch_size,n_dialog,stride,window,shuffle=True):
        split_corpus = [[]]
        with open(path,'r',encoding='utf-8') as f :
            for line in f:
                data=line.strip('\n')
                if(data=='+++$+++'): 
                    split_corpus.append([])
                    if len(split_corpus)+1== n_dialog: break
                else:
                    self.voc.addSentence(data) 
                    split_corpus[-1].append(data)
        f.close()
        self.vocab_size = self.voc.n_words
        c=0
        for dialog in split_corpus:
           if len(dialog) < window + stride:
               split_corpus.remove(dialog)
               continue
           for i in range(0,len(dialog)-window+1,stride):
               local_max = 0
               for j in range(window):
                   local_max+=len(dialog[i+j].split(' '))
               if local_max > c : c = local_max
        self.max_length=c+2
        #print(self.max_length)     

        corpus = []
        for dialog in split_corpus:
            feat = []
            target = []
            for j in range(0,len(dialog)-window+1,stride):
                tmp = ' '.join(dialog[j:j+window])
                feat.append(self.IndexFromSentence(tmp,begin=False,end=False))
                target.append(self.IndexFromSentence(tmp,begin=True,end=True))
            feat = feat[:-1]
            target = target[1:]
            #print('feat = ',len(feat))
            #print('target = ',len(target))
            #input()
            #print('index = ',index,'\n')
            corpus.extend(zip(feat,target))
        self.data_size = len(corpus)
        self.test_data_size = len(corpus)
        dataset=DialogDataset(corpus)
        self.data[name]=DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    def IndexFromSentence(self,s,begin=False,end=True):
        index=[]
        if begin: index.append(self.voc.word2index('SOS'))
        index.extend([self.voc.word2index(word) for word in s.split(' ')])
        if end: index.append(self.voc.word2index('EOS'))
        if len(index)< self.max_length : 
            index.extend([self.voc.word2index('PAD') for i in range(self.max_length - len(index))])
        index = torch.LongTensor(index)
        return index
    def train(self,input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=1):
        encoder.train()
        decoder.train()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = torch.cuda.FloatTensor([0])
        loss_n = 0


        encoder_outputs, encoder_hidden = encoder(input_variable)

        decoder_hidden= decoder.hidden_layer(len(input_variable))
        #decoder_hidden= encoder_hidden
        #print('*'*50)
        #print('target_variable size= ',target_variable.size())
        decoder_input = torch.index_select(target_variable, 1 , Variable(torch.LongTensor([0])).cuda())
        #print('decoder_input size= ',decoder_input.size())
        #print('start decoding ...')
        for di in range(1,self.max_length):
            decoder_output, decoder_hidden= decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                decoder_input=torch.index_select(target_variable, 1, Variable(torch.LongTensor([di])).cuda())
                target=decoder_input.view(-1)
                l,n = self.loss(criterion,decoder_output, target)
                loss = loss + l
                loss_n += n
            else:
                # Without teacher forcing: use its own predictions as the next input
                ni = decoder_output.data.max(1,keepdim=True)[1]
                decoder_input = Variable(ni)
                target=torch.index_select(target_variable, 1, Variable(torch.LongTensor([di])).cuda()).view(-1)
                l,n = self.loss(criterion,decoder_output, target)
                loss = loss + l
                loss_n += n
        #print('finish decoding ...')
        #print('updating parameters ...')
        loss=loss / loss_n
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return float(loss)
    def trainIters(self,encoder, decoder, name, test_name, n_epochs, learning_rate=0.001, print_every=2, plot_every=100):
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        
        criterion = nn.CrossEntropyLoss(size_average=False)
        teacher_forcing_ratio=F.sigmoid(torch.linspace(30,-10,n_epochs))
        for epoch in range(1, n_epochs+ 1):
            start = time.time()
            loss_total=0
            print_loss_total = 0  # Reset every print_every
            for step, (batch_x, batch_y , i) in enumerate(self.data[name]):
                batch_index=step+1
                batch_x=Variable(batch_x).cuda()
                batch_y=Variable(batch_y).cuda()


                loss = self.train(batch_x, batch_y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=teacher_forcing_ratio[epoch])
                # loss
                loss_total+=loss

                if batch_index% print_every == 0:
                    print_loss_avg = (loss_total - print_loss_total )/ print_every
                    print_loss_total = loss_total
                    print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] |  Loss: {:.6f} | Time: {}  '.format(
                                epoch , batch_index*len(batch_x), self.data_size,
                                100. * batch_index*len(batch_x)/ self.data_size, print_loss_avg,
                                self.timeSince(start, batch_index*len(batch_x)/ self.data_size)),end='')
            print('\nTime: {} | Total loss: {:.4f}'.format(self.timeSince(start,1),loss_total/batch_index))
            print('-'*80)
            self.evaluate(encoder,decoder,test_name)
    def evaluate(self,encoder, decoder, name, n=5):
        encoder.eval()
        decoder.eval()

        start = time.time()
        loss=0
        loss_n=0
        decoded_words = []
        record_index = []
        criterion = nn.CrossEntropyLoss(size_average=False)

        print_image=[random.choice(list(self.data[name][0].dataset.feats.keys())) for i in  range(n)]

        data_size = len(self.data[name][0].dataset)
        for step, (batch_x, batch_y , k) in enumerate(self.data[name]):
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
                l, n = self.loss(criterion,decoder_output, target)
                loss += float(l)
                loss_n += n

            words = torch.cat(words,1).unsqueeze(1)
            decoded_words.extend(words)
            record_index.extend(k)
            loss /= loss_n
            print('\rTest (On training set) | [{}/{} ({:.0f}%)] |  Loss: {:.6f} | Time: {}  '.format(
                        batch_index*len(batch_x), data_size,
                        100. * batch_index*len(batch_x)/ data_size, loss,
                        self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')
        
        print('\nTime: {} | Total loss: {:.4f}'.format(self.timeSince(start,1),loss))
        decoded_words = torch.cat(decoded_words,0)
        #print('decoded_words size: ',decoded_words.size())
        for i in print_image:
            #seq_id=self.data[name].dataset.get_id(i)
            seq_list_f = []
            seq_list_d = []
            seq_list_g = []
            for j in self.data[name].dataset.corpus[record_index[i]][0]:
                if j == self.voc.word2index('EOS'): break
                seq_list_f.append(self.voc.index2word[j])
            f_seq = ' ' .join(seq_list_f[1:])
            for j in decoded_words[i]:
                if j == self.voc.word2index('EOS'): break
                seq_list_d.append(self.voc.index2word[j])
            d_seq = ' '.join(seq_list_d)
            for j in self.data[name].dataset.corpus[record_index[i]][1]:
                if j == self.voc.word2index('EOS'): break
                seq_list_g.append(self.voc.index2word[j])
            g_seq = ' '.join(seq_list_g[1:])
            print('i= ',i)
            print('input sequence: {}'.format(f_seq))
            print('decoded_sequence: {}'.format(d_seq))
            print('ground_sequence: {}'.format(g_seq))
        print('-'*60)
 
        return decoded_words
    def loss(self,criterion,output,target):
        check_t=(target!=self.voc.word2index("PAD"))
        t=torch.masked_select(target,check_t).view(-1)
        check_o=check_t.view(-1,1)
        o=torch.masked_select(output,check_o).view(-1,self.vocab_size)
        if len(t)==0: return 0,0
        else : return criterion(o,t),len(t)
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
    def __init__(self,min_count):
        self.w2i= {"SOS":0, "EOS":1, "PAD":2, "UNK":3}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD", 3:"UNK"}
        self.n_words = 4  # Count SOS and EOS and PAD and UNK
        self.min_count=min_count
    def word2index(self,word):
        if word in self.w2i: return self.w2i[word]
        else: return self.w2i["UNK"]
    def addSentence(self, sentence):
        sentence_list = sentence.split()
        for word in sentence_list:
            self.addWord(word)
        max_sen = len(sentence_list)
        return max_sen
    def addWord(self, word):
        word=word.lower()
        if word in self.word2count: self.word2count[word]+=1
        else: self.word2count[word] = 1
        if self.word2count[word] == self.min_count:
            self.w2i[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size , vocab_size , layer_n, dropout=0.3 ):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.hidden = self.initHidden(layer_n)
        self.embedding = nn.Embedding(vocab_size,hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers= layer_n, batch_first=True, dropout=dropout)
    def forward(self, x):
        #x = x.view(-1,28) 
        embedded = self.embedding(x) # (batch_size,max_length,hidden_size)
        #print(embedded.size())
        #input()
        output = embedded
        #print('embedded size = ',embedded.size())
        #print('x len= ',len(x))
        #print('hidden size: ',self.hidden.size())
        hidden = torch.cat([self.hidden for i in range(len(x))],1) # (1,batch_size,hidden_size)
        #print('hidden.size = ',hidden.size())
        output, hidden = self.rnn(output,hidden)
        return output, hidden
    def initHidden(self,layer_n):
        return Variable(torch.zeros(layer_n,1, self.hidden_size),requires_grad=True).cuda()
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, layer_n, hop_n, dropout):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.hop_n= hop_n 
        self.hidden= self.initHidden(layer_n)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.attn = nn.Sequential( nn.Linear(self.hidden_size * (layer_n+1), self.hidden_size),
                    nn.SELU(),
                    nn.Dropout(dropout))
        self.attn_weight = nn.Softmax(1)
        self.attn_combine = nn.Sequential( nn.Linear(self.hidden_size * 2, self.hidden_size))
        self.rnn= nn.GRU(self.hidden_size, self.hidden_size,num_layers= layer_n,batch_first=True, dropout=dropout)
        self.out = nn.Sequential( nn.Linear(self.hidden_size, self.vocab_size),
                    nn.SELU(),
                    nn.Dropout(dropout))
    def forward(self, x, hidden, encoder_outputs):
        # x size: batch * 1
        # encoder outputs size: batch * 80 * hidden
        # hidden size: 1 * batch * hidden
        embedded = self.embedding(x).squeeze(1)         # batch *  hidden

        h=torch.transpose(hidden,0,1).contiguous().view(hidden.size()[1],-1)
        z = self.attn(torch.cat((embedded, h), 1)) # batch * hidden
        # hopping
        for n in range(self.hop_n):
            weight = self.attn_weight(torch.bmm(encoder_outputs,z.unsqueeze(2)).squeeze(2)) # batch * 80 
            z = torch.bmm(weight.unsqueeze(1),encoder_outputs).squeeze(1) # batch * hidden

        output = self.attn_combine(torch.cat((embedded, z), 1)).unsqueeze(1)

        output, hidden=self.rnn(output, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden 
    def hidden_layer(self,n):
        return  torch.cat([self.hidden for i in range(n)],1)
    def initHidden(self,layer_n):
        return Variable(torch.zeros(layer_n,1, self.hidden_size),requires_grad=True).cuda()
class DialogDataset(Dataset):
    def __init__(self,corpus):
        self.corpus = corpus
    #def get_id(self,name):
        #return self.id_[name]
    def __getitem__(self, i):
        x = self.corpus[i][0]
        y = self.corpus[i][1]
        return x,y,i
    def __len__(self):
        return len(self.corpus)

