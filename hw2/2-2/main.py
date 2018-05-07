
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
import sys
assert torch and EncoderRNN and AttnDecoderRNN

EPOCHS = 300
BATCH_SIZE = 128
HIDDEN_LAYER =  1024
NUM_LAYER = 2
NUM_HOP = 3
DROPOUT=0.5
NUM_DIALOG = 1000
#NUM_DIALOG = 56524
STRIDE = 1
WINDOW = 1
MIN_COUNT = 20
MIN_LENGTH = 5

dm = Datamanager(MIN_COUNT)
print('reading data...',end='')
sys.stdout.flush()
dm.get_train_data('train','./data/clr_conversation.txt',batch_size=BATCH_SIZE,n_dialog=NUM_DIALOG,stride=STRIDE,window=WINDOW,shuffle=True)
dm.get_test_data('test','./data/test_input.txt',batch_size=BATCH_SIZE,shuffle=False)
print('\rreading data...finished')
print('Vocabulary size: {}'.format(dm.vocab_size))
print('Max Length: {}'.format(dm.max_length))

encoder=EncoderRNN(HIDDEN_LAYER,dm.vocab_size,NUM_LAYER, DROPOUT).cuda()
#torch.save(encoder,'encoder.pt')
print('finish establish encoder ...')
decoder=AttnDecoderRNN(HIDDEN_LAYER,dm.vocab_size,NUM_LAYER, NUM_HOP, DROPOUT).cuda()
#torch.save(decoder,'decoder.pt')
print('finish establishing decoder ...')

print('start training ...')
dm.trainIters(encoder, decoder, 'train', 'test', EPOCHS)
torch.save(encoder,'encoder.pt')
torch.save(decoder,'decoder.pt')

