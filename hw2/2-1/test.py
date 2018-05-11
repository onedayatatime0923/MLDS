
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
assert torch and EncoderRNN and AttnDecoderRNN

BATCH_SIZE= 128
MAX_LENGTH = 42
TEST_DIR = './data/testing_data/feat'
WRITE_OUTPUT_PATH = './output.txt'
VOLCABULARY_PATH = './vocab.txt'


dm = Datamanager(vocabulary_file= './vocab.txt',max_length= MAX_LENGTH)
#dm.get_data('val','./data/testing_data/feat','./data/testing_label.json','test',batch_size=BATCH_SIZE,shuffle=False)
dm.get_test_data('test', TEST_DIR, batch_size=BATCH_SIZE,shuffle=False)
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

#dm.evaluate(encoder, decoder, 'val')
dm.predict(encoder, decoder, 'test', write_file=WRITE_OUTPUT_PATH)

