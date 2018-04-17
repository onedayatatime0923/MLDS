
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
assert torch and EncoderRNN and AttnDecoderRNN

EPOCHS=20
BATCH_SIZE=256
HIDDEN_LAYER=512


dm = Datamanager()
dm.get_data('train','./data/training_data/feat','./data/training_label.json',batch_size=BATCH_SIZE)

encoder=EncoderRNN(4096,HIDDEN_LAYER).cuda()
torch.save(encoder,'encoder.pt')
decoder=AttnDecoderRNN(HIDDEN_LAYER,dm.vocab_size,80).cuda()
torch.save(decoder,'decoder.pt')

dm.trainIters(encoder, decoder, 'train', EPOCHS)
torch.save(encoder,'encoder.pt')
torch.save(decoder,'decoder.pt')

