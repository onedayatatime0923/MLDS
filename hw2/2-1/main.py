
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
assert torch and EncoderRNN and AttnDecoderRNN

EPOCHS=300
BATCH_SIZE=256
HIDDEN_LAYER=512
N_LAYER=3
DROPOUT=0.3


dm = Datamanager()
dm.get_data('train','../data/training_data/feat','../data/training_label.json','train',batch_size=BATCH_SIZE)
dm.get_data('test','../data/testing_data/feat','../data/testing_label.json','test',batch_size=BATCH_SIZE,shuffle=False)

encoder=EncoderRNN(4096,HIDDEN_LAYER,N_LAYER,dropout=DROPOUT).cuda()
torch.save(encoder,'encoder.pt')
decoder=AttnDecoderRNN(HIDDEN_LAYER,dm.vocab_size,N_LAYER,dropout=DROPOUT).cuda()
torch.save(decoder,'decoder.pt')

dm.trainIters(encoder, decoder, 'train', 'test', EPOCHS, 'data/output.txt')
torch.save(encoder,'encoder.pt')
torch.save(decoder,'decoder.pt')

