
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
assert torch and EncoderRNN and AttnDecoderRNN

EPOCHS = 300
BATCH_SIZE = 32
HIDDEN_LAYER = 128
NUM_DIALOG = 8000
NUM_LAYER = 1
STRIDE = 3
WINDOW = 3
MIN_COUNT = 3

dm = Datamanager(MIN_COUNT)
dm.get_data('train','clr_conversation.txt','train',batch_size=BATCH_SIZE,dialog=NUM_DIALOG,stride=STRIDE,window=WINDOW,shuffle=True)
#dm.get_data('test','./data/testing_data/feat','./data/testing_label.json','test',batch_size=BATCH_SIZE,shuffle=False)
print('finish data processing ...')

encoder=EncoderRNN(HIDDEN_LAYER,HIDDEN_LAYER,dm.vocab_size,NUM_LAYER).cuda()
torch.save(encoder,'encoder.pt')
print('finish establish encoder ...')
decoder=AttnDecoderRNN(HIDDEN_LAYER,dm.vocab_size,NUM_LAYER).cuda()
torch.save(decoder,'decoder.pt')
print('finish establishing decoder ...')

print('start training ...')
dm.trainIters(encoder, decoder, 'train', 'train', EPOCHS)
torch.save(encoder,'encoder.pt')
torch.save(decoder,'decoder.pt')

