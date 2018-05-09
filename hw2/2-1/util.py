
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch import optim
import os
import numpy as np
from functools import reduce 
import operator
import json
import random
import time
import math
import matplotlib.pyplot as plt
assert os and np and F and math


class Datamanager:
    def __init__(self,min_count):
        self.voc=Vocabulary(min_count)
        self.data={}
        self.vocab_size=0
        self.max_length=0
    def get_data(self,name,f_path,l_path,mode,batch_size,shuffle=True):
        # self.data[name]=[ dataloader, labels]
        feats={}
        captions_id={}
        captions_str={}
        max_sen=0
        for i in os.listdir(f_path):
            if not i.startswith('.'):
                x=torch.FloatTensor(np.load('{}/{}'.format(f_path,i)))
                feats[i[:-4]]=x

        with open(l_path) as f:
            labels=json.load(f)
        if mode== 'train':
            for l in labels:
                m=self.voc.addSentence(l['caption'])
                if m>max_sen: max_sen=m
            self.max_length=max_sen+2
            self.vocab_size=self.voc.n_words
            # save the captions_str is for getting the grounded sequence when evaluating
            for l in labels:
                captions_id[l['id']]=self.IndexFromSentence(l['caption'],begin=True,end=True)
                captions_str[l['id']]=[x.rstrip('.') for x in l['caption']]
        elif mode== 'test':
            # save the captions_str is for getting the grounded sequence when evaluating
            for l in labels:
                captions_id[l['id']]=self.IndexFromSentence([l['caption'][0]],begin=True,end=True)
                captions_str[l['id']]= [x.rstrip('.') for x in l['caption']]
        else : raise ValueError('Wrong mode.')
        dataset=VideoDataset(feats,captions_id)
        self.data[name]= [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), captions_str]
    def IndexFromSentence(self,sentences,begin=False,end=True):
        indexes=[]
        for s in sentences:
            index=[]
            if begin: index.append(self.voc.word2index('SOS'))
            index.extend([self.voc.word2index(word) for  word in s.split(' ')])
            if end: index.append(self.voc.word2index('EOS'))
            if len(index)< self.max_length : 
                index.extend([self.voc.word2index('PAD') for i in range(self.max_length  -len(index))])
            indexes.append(index)
        indexes = torch.LongTensor(indexes)
        return indexes
    def train(self,input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, words, teacher_forcing_ratio=1):
        encoder.train()
        decoder.train()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = Variable(torch.cuda.FloatTensor([0]).cuda())
        loss_n = 0

        encoder_outputs, encoder_hidden = encoder(input_variable)


        #decoder_hidden= decoder.hidden_layer(len(input_variable))
        decoder_hidden= encoder_hidden
        decoder_input=torch.index_select(target_variable, 1, Variable(torch.LongTensor([0])).cuda())
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        for di in range(1,self.max_length):
            decoder_output, decoder_hidden= decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                decoder_input=torch.index_select(target_variable, 1, Variable(torch.LongTensor([di])).cuda())
                target=decoder_input.view(-1)
                l,n = self.loss(criterion,decoder_output, target)
                loss = loss + l
                loss_n += n
                words.append(decoder_output.data.max(1,keepdim=True)[1])
            else:
                # Without teacher forcing: use its own predictions as the next input
                ni = decoder_output.data.max(1,keepdim=True)[1]
                decoder_input = Variable(ni)
                target=torch.index_select(target_variable, 1, Variable(torch.LongTensor([di])).cuda()).view(-1)
                l,n = self.loss(criterion,decoder_output, target)
                loss = loss + l
                loss_n += n
                words.append(ni)

        loss=loss / loss_n
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return float(loss)/ (self.max_length)
    def trainIters(self,encoder, decoder, name, test_name, n_epochs, write_file, plot_file, learning_rate=0.001, print_every=2):
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        
        criterion = nn.CrossEntropyLoss(size_average=False)
        teacher_forcing_ratio=F.sigmoid(torch.linspace(30,-5,n_epochs))
        data_size = len(self.data[name][0].dataset)
        record=0
        loss_bleu_list=[]
        for epoch in range(n_epochs):
            start = time.time()
            loss_total=0
            print_loss_total = 0  # Reset every print_every
            bleu=[]
            for step, (batch_x, batch_y, video) in enumerate(self.data[name][0]):
                batch_index=step+1
                batch_x=Variable(batch_x).cuda()
                batch_y=Variable(batch_y).cuda()
                words=[]

                loss = self.train(batch_x, batch_y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, words, teacher_forcing_ratio=teacher_forcing_ratio[epoch])
                # loss
                loss_total+=loss
                # bleu
                words= torch.cat(words,1)
                bleu.extend(self.bleu_batch(words,name,video[0]))

                if batch_index% print_every == 0:
                    print_loss_avg = (loss_total - print_loss_total )/ print_every
                    print_loss_total = loss_total
                    print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] |  Loss: {:.6f} | Time: {}  '.format(
                                epoch+1 , batch_index*len(batch_x), data_size,
                                100. * batch_index*len(batch_x)/ data_size, print_loss_avg,
                                self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')
            bleu_average = sum(bleu) / len(bleu)
            print('\nTime: {} | Total loss: {:.4f} | Bleu Score: {:.5f}'.format(self.timeSince(start,1),loss_total/batch_index,bleu_average))
            print('-'*80)
            if epoch%10==0: self.evaluate(encoder,decoder,name, n=3)
            #record=self.evaluate(encoder,decoder,test_name, write_file, record, n=5)
            self.evaluate(encoder,decoder,test_name, write_file, record, n=5)
            loss_bleu_list.append([loss_total/ batch_index, bleu_average])
            self.plot(loss_bleu_list, plot_file)
    def evaluate(self,encoder, decoder, name, write_file=None, record=0, n=5):
        encoder.eval()
        decoder.eval()

        start = time.time()
        loss=0
        loss_n=0
        decoded_words = []
        videos = [[],[]]
        criterion = nn.CrossEntropyLoss(size_average=False)

        print_image=[random.choice(list(self.data[name][0].dataset.feats.keys())) for i in  range(n)]

        data_size = len(self.data[name][0].dataset)
        for step, (batch_x, batch_y,video) in enumerate(self.data[name][0]):
            batch_index=step+1
            batch_x=Variable(batch_x).cuda()
            batch_y=Variable(batch_y).cuda()

            encoder_outputs, encoder_hidden = encoder(batch_x)

            decoder_hidden= encoder_hidden
            decoder_input=torch.index_select(batch_y, 1, Variable(torch.LongTensor([0])).cuda())

            words=[]
            bleu=[]

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

            words= torch.cat(words,1)
            bleu.extend(self.bleu_batch(words, name, video[0]))

            decoded_words.extend(words.unsqueeze(1))
            videos[0].extend(video[0])
            videos[1].extend(video[1])

            loss /= loss_n

            print('\r{} | [{}/{} ({:.0f}%)] |  Loss: {:.6f} | Time: {}  '.format(
                        name.upper(),
                        batch_index*len(batch_x), data_size,
                        100. * batch_index*len(batch_x)/ data_size, loss,
                        self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')

        bleu_average = sum(bleu) / len(bleu)
        decoded_words=torch.cat(decoded_words,0)
        print('\nTime: {} | Bleu Score: {:.5f}'.format(self.timeSince(start,1),bleu_average))
        # output decoded and ground sequence
        for i in print_image:
            seq_id=videos[0].index(i)
            seq_list=[]
            for j in decoded_words[seq_id]:
                index=int(j)
                if index ==self.voc.word2index('EOS'): break
                seq_list.append(self.voc.index2word[index])
            d_seq=' '.join(seq_list)
            g_seq=self.data[name][1][i][videos[1][seq_id]]
            print('id: {:<25} | decoded_sequence: {}'.format(i,d_seq))
            print('    {:<25} | ground_sequence: {}'.format(' '*len(i),g_seq))
        # writing output file
        if write_file!=None and bleu_average > record:
            self.write(write_file,decoded_words,name,video[0])
        print('-'*80)
        if bleu_average>record: return bleu_average
        else: return record
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
    def write(self,path,decoded_words,name,video):
        with open(path,'w') as f:
            for i in range(len(video)):
                seq_list=[]
                for j in decoded_words[i]:
                    index=int(j)
                    if index ==self.voc.word2index('EOS'): break
                    seq_list.append(self.voc.index2word[index])
                d_seq=' '.join(seq_list)
                f.write('{},{}\n'.format(video[i],d_seq))
    def bleu_batch(self, words, name, video):
        bleu=[]
        for i in range(len(video)):
            seq_list=[]
            for j in words[i]:
                index=int(j)
                if index ==self.voc.word2index('EOS'): break
                seq_list.append(self.voc.index2word[index])
            seq_list=' '.join(seq_list)
            target_list=self.data[name][1][video[i]]
            if (len(seq_list)!=0):
                bleu.append(self.BLEU(seq_list,target_list,True))
        return bleu
    def BLEU(self,s,t,flag = False):
        score = 0.  
        candidate = [s.strip()]
        if flag:
            references = [[t[i].strip()] for i in range(len(t))]
        else:
            references = [[t.strip()]] 
        precisions = []
        pr, bp = self.count_ngram(candidate, references, 1)
        precisions.append(pr)
        score = self.geometric_mean(precisions) * bp
        return score
    def geometric_mean(self,precisions):
        return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))
    def brevity_penalty(self,c, r):
        if c > r:
            bp = 1
        else:
            bp = math.exp(1-(float(r)/c))
        return bp
    def best_length_match(self,ref_l, cand_l):
        """Find the closest length of reference to that of candidate"""
        least_diff = abs(cand_l-ref_l[0])
        best = ref_l[0]
        for ref in ref_l:
            if abs(cand_l-ref) < least_diff:
                least_diff = abs(cand_l-ref)
                best = ref
        return best
    def count_ngram(self,candidate, references, n):
        clipped_count = 0
        count = 0
        r = 0
        c = 0
        for si in range(len(candidate)):
            # Calculate precision for each sentence
            ref_counts = []
            ref_lengths = []
            # Build dictionary of ngram counts
            for reference in references:
                ref_sentence = reference[si]
                ngram_d = {}
                words = ref_sentence.strip().split()
                ref_lengths.append(len(words))
                limits = len(words) - n + 1
                # loop through the sentance consider the ngram length
                for i in range(limits):
                    ngram = ' '.join(words[i:i+n]).lower()
                    if ngram in ngram_d.keys():
                        ngram_d[ngram] += 1
                    else:
                        ngram_d[ngram] = 1
                ref_counts.append(ngram_d)
            # candidate
            cand_sentence = candidate[si]
            cand_dict = {}
            words = cand_sentence.strip().split()
            limits = len(words) - n + 1
            for i in range(0, limits):
                ngram = ' '.join(words[i:i + n]).lower()
                if ngram in cand_dict:
                    cand_dict[ngram] += 1
                else:
                    cand_dict[ngram] = 1
            clipped_count += self.clip_count(cand_dict, ref_counts)
            count += limits
            r += self.best_length_match(ref_lengths, len(words))
            c += len(words)
        if clipped_count == 0:
            pr = 0
        else:
            pr = float(clipped_count) / count
        bp = self.brevity_penalty(c, r)
        return pr, bp
    def clip_count(self,cand_d, ref_ds):
        """Count the clip count for each ngram considering all references"""
        count = 0
        for m in cand_d.keys():
            m_w = cand_d[m]
            m_max = 0
            for ref in ref_ds:
                if m in ref:
                    m_max = max(m_max, ref[m])
            m_w = min(m_w, m_max)
            count += m_w
        return count
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def plot(self, record, path):
        x=np.array(list(range(1,len(record)+1)),dtype=np.uint8)
        y=np.array(record)
        plt.figure()
        plt.plot(x,y[:,0],'b',label='loss')
        plt.plot(x,y[:,1],'g',label='bleu')
        plt.legend()
        plt.savefig(path)
        plt.close()
class Vocabulary:
    def __init__(self,min_count):
        self.w2i= {"SOS":0, "EOS":1, "PAD":2, "UNK":3}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD", 3:"UNK"}
        self.n_words = 4  # Count SOS and EOS and PAD and UNK
        self.min_count=min_count
    def word2index(self,word):
        word=word.lower()
        if word in self.w2i: return self.w2i[word]
        else: return self.w2i["UNK"]
    def addSentence(self, sentences):
        max_sen=0
        for sentence in sentences:
            sentence_list=sentence.split(' ')
            for word in sentence_list:
                self.addWord(word)
            if len(sentence_list)>max_sen: max_sen=len(sentence_list)
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
    def __init__(self,input_size, hidden_size, layer_n, dropout=0.3):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.hidden = self.initHidden(layer_n)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers= layer_n, batch_first=True, dropout=dropout)
    def forward(self, x):
        hidden = torch.cat([self.hidden for i in range(len(x))],1)
        output, hidden = self.rnn(x, hidden)
        output = output / torch.matmul(torch.norm(output,2,dim=2).unsqueeze(2),Variable(torch.ones(1,self.hidden_size)).cuda())
        return output,  hidden
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
class VideoDataset(Dataset):
    def __init__(self, feats, captions):
        self.feats=feats
        self.captions=captions
        index=[]
        for i in captions:
            index.extend([(i,j) for j in range(len(captions[i]))])
        self.index=index
        #print(len(index))
    def __getitem__(self, i):
        x=self.feats[self.index[i][0]]
        #x+=torch.normal(torch.zeros_like(x),0.1)
        y=self.captions[self.index[i][0]][self.index[i][1]]
        return x,y,self.index[i]
    def __len__(self):
        return len(self.index)
