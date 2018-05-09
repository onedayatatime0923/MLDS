
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
import sys
assert torch and EncoderRNN and AttnDecoderRNN

EPOCHS = 200
BATCH_SIZE = 100
HIDDEN_LAYER = 1024
NUM_LAYER = 2
NUM_HOP = 3
DROPOUT = 0.5
NUM_DIALOG = 10000
NUM_DIALOG = 56524
NUM_PAIR = 65536
NUM_PAIR = None
MAX_LENGTH = 15
MIN_LENGTH = 2
MIN_COUNT = 10

dm = Datamanager(MIN_COUNT, MAX_LENGTH, MIN_LENGTH)
print('reading data...')
sys.stdout.flush()
dm.get_train_data('train','./data/clr_conversation.txt',n_dialog=NUM_DIALOG, n_pair= NUM_PAIR, batch_size= BATCH_SIZE, shuffle=True)
dm.get_test_data('test','./data/test_input.txt',batch_size=BATCH_SIZE,shuffle=False)
print('\rreading data...finished')
print('Vocabulary size: {}'.format(dm.vocab_size))
print('Max Length: {}'.format(dm.max_len))
#input()

encoder=EncoderRNN(HIDDEN_LAYER,dm.vocab_size,NUM_LAYER, DROPOUT).cuda()
#torch.save(encoder,'encoder.pt')
print('finish establishing encoder ...')
decoder=AttnDecoderRNN(HIDDEN_LAYER,dm.vocab_size,NUM_LAYER, NUM_HOP, DROPOUT).cuda()
#torch.save(decoder,'decoder.pt')
print('finish establishing decoder ...')

print('start training ...')
dm.trainIters(encoder, decoder, 'train', 'test', EPOCHS)
torch.save(encoder,'encoder.pt')
torch.save(decoder,'decoder.pt')

