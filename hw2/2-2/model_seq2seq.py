from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
import sys
import os 
assert torch and EncoderRNN and AttnDecoderRNN

BATCH_SIZE = 100
HIDDEN_LAYER = 1024
NUM_LAYER = 3
NUM_HOP = 3
DROPOUT = 0.5
NUM_DIALOG = 10000
NUM_DIALOG = 56524
NUM_PAIR = None
NUM_PAIR = 65536
MAX_LENGTH = 15
MIN_LENGTH = 2
MIN_COUNT = 10
#OUTPUT_FILE = './output.txt'
VOLCABULARY_PATH = './vocab.txt'
inputdata = sys.argv[3]
outputdata = sys.argv[4]

dm = Datamanager(MIN_COUNT, MAX_LENGTH, MIN_LENGTH,VOLCABULARY_PATH)
print('reading data...')
sys.stdout.flush()
dm.get_test_data('test',inputdata,batch_size=BATCH_SIZE,shuffle=False)
print('\rreading data...finished')

encoder = torch.load(sys.argv[1])
decoder = torch.load(sys.argv[2])

dm.test(encoder,decoder,'test',outputdata)










