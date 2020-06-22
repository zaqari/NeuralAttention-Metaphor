import torch
import numpy as np
torch.autograd.set_detect_anomaly(True)

file_path = '/home/zro/d/s-0528/'
data_source = '-t2'
session_path = '/home/zro/d/checkpoints/sessions/s/'

pretrained = False
saveStates = True
iEpochs = 20            #set if pretrained=False to 10
continueTraining = 10   #set if continuing training from initial save

#decoder_col_list = ['tref.stem', 'tref.dep', 'head.stem']
decoder_col_list = ['lex']
device = torch.device('cpu')

print("""NOTE: for the datasaver module, I temporarily commented out the line for m.out.eval()""")





################################################################
########### Building lexical/w2v representations
################################################################
head_list_restrictions = ['victori','reduc','advanc','kill','mountain','allevi','worship','defend','win']

from kgen2.lexemes import *

enc_all = pd.read_csv(file_path+'enc.csv')
dec_all = pd.read_csv(file_path+'dec.csv')

print(list(dec_all))
#dec_all = dec_all.drop_duplicates(subset=['tref.stem', 'tref.dep', 'head.stem', 'sent'])
#dec_all.index=range(len(dec_all))
#restriction = dec_all.loc[dec_all['span'].isin([1., '1']) & dec_all['head.stem'].isin(head_list_restrictions)]

train_sents = pd.read_csv(file_path+'train.csv')
test_sents = pd.read_csv(file_path+'test.csv')
#extract_sents = pd.read_csv(file_path+'extract-spans-tra.csv')

train_sents = train_sents['sent'].unique().tolist() #[sent for sent in train_sents['sent'].unique() if sent in restriction['sent'].unique()]

test_sents = test_sents['sent'].unique().tolist() #[sent for sent in test_sents['sent'].unique() if sent in restriction['sent'].unique()] #list(dec_all['sent'].loc[dec_all['span'].isin([1.])].unique())

print('@df.cmsource.len: {} '.format(len(dec_all['cmsource'].loc[dec_all['sent'].isin(train_sents+test_sents)].unique())))

data_sents = train_sents+test_sents
data_sents = list(set(data_sents))

print('@df[ {} ].len: {} sentences'.format(file_path, len(data_sents)))





################################################################
########### Setting up input dictionaries and embedding layers
################################################################
lex = lexemes()
tar = lexemes()

f2id = None
t2id = None

if pretrained:
    #If loading an existing session dictionary
    f2id = lex.load_from_file(session_path+'f2id.csv')
    t2id = tar.load_from_file(session_path+'t2id.csv')

else:
    #If creating a dictionary from scratch
    f2id = lex.id_dic(enc_all.loc[enc_all['sent'].isin(data_sents)], ['lex'])
    t2id = tar.id_dic(dec_all.loc[dec_all['sent'].isin(data_sents)], decoder_col_list)
    lex.dic_persist(f2id, session_path+'f2id.csv')
    tar.dic_persist(t2id, session_path+'t2id.csv')

#Pulling pretrained embeddings from file
f_vecs, f_err = lex.embeds(f2id)
t_vecs, t_err = tar.embeds(t2id)





################################################################
########### Setting up inputs to be passed to NN
################################################################
from kgen2.torchNN.inputs import *
encX = cData(enc_all.loc[enc_all['sent'].isin(data_sents)], 'lex', f2id, data_sents)
#decX = rData(dec_all, decoder_col_list, t2id, data_sents, SOS_token=True)
decX = cData(dec_all, decoder_col_list[0], t2id, data_sents, SOS_token=True)
dIDX = cIndx(dec_all, [['head', 'lex'], ['tref', 'lex']], data_sents)
print('decoder columns: {} \n'.format(decoder_col_list))





################################################################
########### Setting up outputs to be passed to NN
################################################################
from kgen2.torchNN.outputs import *
#decY = rEVecs(dec_all, decoder_col_list, t2id, data_sents, t_vecs)
decY = cVecs(dec_all, decoder_col_list, t2id, data_sents, t_vecs, SOS_token=True)#rRLVecs2(dec_all, decoder_col_list, t2id, data_sents, t_vecs, train_sents, SOS_token=True)
#decY = cVecs(dec_all, 'lex', t2id, data_sents, t_vecs)





################################################################
########### Set up layers and main model
################################################################
from kgen2.torchNN.NNs.gru3.encoder import *
from kgen2.torchNN.NNs.gru3.decoder import *
from kgen2.torchNN.NNs.gru3.attention2 import *
from kgen2.torchNN.NNs.gru3.outlayer import *
from kgen2.torchNN.NNs.gru3.mRL2 import *
from kgen2.datasaverDeepMem import *

encoder = enc(device=device, embeddings=f_vecs, bidirectional=False, hidden_size=600)
attention = rosen(key_size=encoder.output_size, query_size=encoder.output_size, hidden_size=600)
global_memory = cosR5(key_size=600, query_size=600, hidden_size=600, novelty_bias=.5)
outout = tanH_out(hidden_size=encoder.output_size, output_size=300)
decoder = dec(device=device, attention=attention, outputlayer=outout, embeddings=t_vecs, context_size=encoder.output_size, bidirectional=False, hidden_size=600, SOS_token=True)
decoder.n_classes = 300

m = model(encoder, decoder, global_memory, memory_size=(len(data_sents), len(decoder_col_list)-1, 600)).cpu()






################################################################
########### Save function to save on time & space
################################################################
def save(filename, length=len(decoder_col_list)):
    if len(decoder_col_list) > 1:
        _ = rSave(m, encX, decX, data_sents, dec_all,
                                  attn_data=(session_path + 'attention-{}.csv'.format(filename), enc_all, decoder_col_list),
                                  states_out=session_path + 'states-{}.csv'.format(filename),
                                  statesize=length * 600
                                  )

    else:
        _ = cSave(m, encX, decX, data_sents, dec_all,
                                  attn_data=(session_path + 'attention-{}.csv'.format(filename), enc_all, decoder_col_list[0]),
                                  states_out=session_path + 'states-{}.csv'.format(filename),
                                  )





################################################################
########### Training Regimen
################################################################
if pretrained:
    m.enc.load_state_dict(torch.load(session_path + 'encoder-2.pt'))
    m.dec.attention.load_state_dict(torch.load(session_path + 'localattn-2.pt'))
    m.dec.load_state_dict(torch.load(session_path + 'decoder-2.pt'))
    m.dec.outlayer.load_state_dict(torch.load(session_path + 'output-2.pt'))

    print('model evaluation step')
    m.evaluation(encX, decX, decY, test_sents,
                 nn.CosineEmbeddingLoss())
    save('ep{}'.format(iEpochs))
    print('pretrained model loaded\n')

else:
    optimizer = torch.optim.Adam([{'params': encoder.parameters()},
                                   {'params': decoder.parameters()},
                                   #{'params': outout.parameters()},
                                   #{'params': global_memory.parameters(), 'lr': .05}
                                  ], lr=0.0001)
    m.batched_training(encX, decX, decY, train_sents, optimizer,
                       nn.CosineEmbeddingLoss(), #change to nn.MSELoss()?
                       validation_data=(encX, decX, decY, test_sents),
                       pull_from_memory=False,
                       epochs=iEpochs,
                       cutoff=.01)

    torch.save(m.enc.state_dict(), session_path+'encoder.pt')
    torch.save(m.dec.state_dict(), session_path+'decoder.pt')
    torch.save(m.dec.attention.state_dict(), session_path+'localattn.pt')
    torch.save(m.dec.outlayer.state_dict(), session_path+'output.pt')

    if saveStates:
        save('ep{}'.format(iEpochs))

    print('initial training complete\n')





################################################################
########### Continue training from save point
################################################################
if continueTraining:
    optimizer = torch.optim.Adam([{'params': encoder.parameters()},
                                  {'params': decoder.parameters()},
                                  #{'params': outout.parameters()},
                                  # {'params': global_memory.parameters(), 'lr': .05}
                                  ], lr=0.0001)

    m.batched_training(encX, decX, decY, train_sents, optimizer,
                       nn.CosineEmbeddingLoss(),
                       validation_data=(encX, decX, decY, test_sents),
                       pull_from_memory=False,
                       epochs=continueTraining,
                       cutoff=.01)

    torch.save(m.enc.state_dict(), session_path + 'encoder-2.pt')
    torch.save(m.dec.state_dict(), session_path + 'decoder-2.pt')
    torch.save(m.dec.attention.state_dict(), session_path + 'localattn-2.pt')
    torch.save(m.dec.outlayer.state_dict(), session_path + 'output-2.pt')

    if saveStates:
        save('ep{}'.format(continueTraining+iEpochs))
