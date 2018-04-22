
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
assert torch and EncoderRNN and AttnDecoderRNN

EPOCHS = 300
BATCH_SIZE = 128
HIDDEN_LAYER = 128


dm = Datamanager()
dm.get_data('train','clr_conversation.txt','train',batch_size=BATCH_SIZE,shuffle=False)
#dm.get_data('test','./data/testing_data/feat','./data/testing_label.json','test',batch_size=BATCH_SIZE,shuffle=False)
print('finish data processing ...')

encoder=EncoderRNN(HIDDEN_LAYER,HIDDEN_LAYER,dm.vocab_size).cuda()
torch.save(encoder,'encoder.pt')
print('finish establish encoder ...')
decoder=AttnDecoderRNN(HIDDEN_LAYER,dm.vocab_size).cuda()
torch.save(decoder,'decoder.pt')
print('finish establishing decoder ...')

print('start training ...')
dm.trainIters(encoder, decoder, 'train', 'train', EPOCHS)
torch.save(encoder,'encoder.pt')
torch.save(decoder,'decoder.pt')

