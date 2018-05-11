
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
import sys
import os 
assert torch and EncoderRNN and AttnDecoderRNN

EPOCHS = 150
BATCH_SIZE = 100
HIDDEN_LAYER =  512
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
OUTPUT_FILE = './output.txt'
VOLCABULARY_PATH = None #'./vocab.txt'

dm = Datamanager(MIN_COUNT, MAX_LENGTH, MIN_LENGTH,VOLCABULARY_PATH)
print('reading data...')
sys.stdout.flush()
dm.get_train_data('train','./data/clr_conversation.txt',n_dialog=NUM_DIALOG, n_pair= NUM_PAIR, batch_size= BATCH_SIZE, shuffle=True)
dm.get_test_data('test','./data/test_input.txt',batch_size=BATCH_SIZE,shuffle=False)
print('\rreading data...finished')
#dm.voc.save(VOLCABULARY_PATH)
#print('Vocabulary size: {}'.format(dm.vocab_size))
#print('Max Length: {}'.format(dm.max_len))
#input()

#encoder=EncoderRNN(HIDDEN_LAYER,dm.vocab_size,NUM_LAYER, DROPOUT).cuda()
#print('finish establishing encoder ...')
#decoder=AttnDecoderRNN(HIDDEN_LAYER,dm.vocab_size,NUM_LAYER, NUM_HOP, DROPOUT).cuda()
#print('finish establishing decoder ...')

#print('start training ...')
#dm.trainIters(encoder, decoder, 'train', 'test', EPOCHS, output_path = OUTPUT_FILE)
#torch.save(encoder,'encoder.pt')
#torch.save(decoder,'decoder.pt')

encoder = torch.load(sys.argv[1])
decoder = torch.load(sys.argv[2])

dm.test(encoder,decoder,'test','../../output.txt')

os.chdir('data/evaluation')
os.system('python main.py ../test_input.txt ../../output.txt')









