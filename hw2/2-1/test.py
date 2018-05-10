
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
assert torch and EncoderRNN and AttnDecoderRNN

EPOCHS=50
BATCH_SIZE= 128
HIDDEN_LAYER= 1024
LAYER_N=3
HOP_N=3
DROPOUT=0.5
MAX_LENGTH = 42
MIN_COUNT = 3
WRITE_OUTPUT_PATH = './output.txt'
VOLCABULARY_PATH = './vocab.txt'


dm = Datamanager(MIN_COUNT)
dm.voc.load(VOLCABULARY_PATH)
dm.get_test_data('test','./data/testing_data/feat', max_length= MAX_LENGTH, batch_size=BATCH_SIZE,shuffle=False)
print('Max length: {}'.format(dm.max_length))
print('Vocabulary size: {}'.format(dm.voc.n_words))

#encoder=EncoderRNN(4096,HIDDEN_LAYER, LAYER_N,dropout=DROPOUT).cuda()
#decoder=AttnDecoderRNN(HIDDEN_LAYER,dm.vocab_size, LAYER_N, HOP_N, dropout=DROPOUT).cuda()
encoder=torch.load('encoder.pt')
decoder=torch.load('decoder.pt')
print("Encoder Parameter: {}".format(dm.count_parameters(encoder)))
print("Decoder Parameter: {}".format(dm.count_parameters(decoder)))
#torch.save(encoder,'encoder.pt')
#torch.save(decoder,'decoder.pt')

dm.predict(encoder, decoder, 'test', write_file=WRITE_OUTPUT_PATH)

