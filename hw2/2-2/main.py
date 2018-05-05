
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
assert torch and EncoderRNN and AttnDecoderRNN

EPOCHS = 300
BATCH_SIZE = 256
HIDDEN_LAYER =  256
NUM_LAYER = 2
NUM_HOP = 3
DROPOUT=0.3
NUM_DIALOG = 8000
NUM_DIALOG = 56524
STRIDE = 3
WINDOW = 3
MIN_COUNT = 3

dm = Datamanager(MIN_COUNT)
print('reading data...',end='')
dm.get_data('train','clr_conversation.txt',batch_size=BATCH_SIZE,n_dialog=NUM_DIALOG,stride=STRIDE,window=WINDOW,shuffle=True)
print('\rreading data...finished')
print('Vocabulary size: {}'.format(dm.vocab_size))
input()

encoder=EncoderRNN(HIDDEN_LAYER,dm.vocab_size,NUM_LAYER, DROPOUT).cuda()
#torch.save(encoder,'encoder.pt')
print('finish establish encoder ...')
decoder=AttnDecoderRNN(HIDDEN_LAYER,dm.vocab_size,NUM_LAYER, NUM_HOP, DROPOUT).cuda()
#torch.save(decoder,'decoder.pt')
print('finish establishing decoder ...')

print('start training ...')
dm.trainIters(encoder, decoder, 'train', 'train', EPOCHS)
torch.save(encoder,'encoder.pt')
torch.save(decoder,'decoder.pt')

