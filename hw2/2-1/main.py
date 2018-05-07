
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
assert torch and EncoderRNN and AttnDecoderRNN

EPOCHS=2
BATCH_SIZE= 512
HIDDEN_LAYER= 128
LAYER_N=1
HOP_N=1
DROPOUT=0.5
MIN_COUNT = 3
OUTPUT_PATH = './data/record.png'


dm = Datamanager(MIN_COUNT)
dm.get_data('train','./data/training_data/feat','./data/training_label.json','train',batch_size=BATCH_SIZE)
dm.get_data('test','./data/testing_data/feat','./data/testing_label.json','test',batch_size=BATCH_SIZE,shuffle=False)

encoder=EncoderRNN(4096,HIDDEN_LAYER, LAYER_N,dropout=DROPOUT).cuda()
decoder=AttnDecoderRNN(HIDDEN_LAYER,dm.vocab_size, LAYER_N, HOP_N, dropout=DROPOUT).cuda()
print("Encoder Parameter: {}".format(dm.count_parameters(encoder)))
print("Decoder Parameter: {}".format(dm.count_parameters(decoder)))
#torch.save(encoder,'encoder.pt')
#torch.save(decoder,'decoder.pt')

record=dm.trainIters(encoder, decoder, 'train', 'test', EPOCHS, 'data/output.txt')
torch.save(encoder,'encoder.pt')
torch.save(decoder,'decoder.pt')
dm.plot(record,OUTPUT_PATH)

