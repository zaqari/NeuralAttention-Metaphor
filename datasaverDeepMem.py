import pandas as pd
import numpy as np
import torch

def rSave(m, encX, decX, test_sents, dec_all, states_out, statesize=1200, state_relational_columns=['sentID', 'tref.stem', 'head.stem', 'tref.dep', 'cmsource', 'span'], attn_data=None, outputs_out=None):
    """

    :param m: the model being used to generate our interpretations.
    :param encX: encoder data
    :param decX: decoder data
    :param test_sents: the sentence labels for the sentences that we're going to be testing on
    :param enc_all: the originally imported document for our encoder data. Even post conversion, it still holds a lot of really useful info.
    :param dec_all: the originally imported document for our decoder data. Even post conversion, it still holds a lot of really useful info.
    :param attn_data: a list or tuple containing at index0 an output file location, index1 the encoder data used to find qualitative weights, and optional index2 a list of cols used for steps.
    :param horizontal: Occasionally, our values for decoder steps are going to be horizontal, per sentence. This triggers that.
    :param streaming: must include 'encoder_df': encoder_df, 'decoder_fx': decoder_df, 'encoder_fx': (function, cols), 'decoder_fx': (function, cols)
    :return:
    """
    m.enc.eval()
    m.dec.eval()
    m.gATTN.eval()
    #m.out.eval()

    alpha_df = None
    if attn_data:
        alpha_df = pd.DataFrame(columns=['sent', 'cm', 'step', 'weights', 'topN'])
        alpha_df.to_csv(attn_data[0], index=False, encoding='utf-8')

    hidden_df = pd.DataFrame(columns=['sent']+[str(i) for i in range(statesize)])
    hidden_df.to_csv(states_out, index=False, encoding='utf-8')

    outputs_df = None
    if outputs_out:
        outputs_df = pd.DataFrame(columns=['sent', 'tag1', 'tag2', 'tag3']) #else columns=['sent', 'tag']
        outputs_df.to_csv(outputs_out, index=False, encoding='utf-8')

    #If for some reason you have some subset of sentences in the
    # data that you wish to run testing procedures on, besides
    # the cummulative data, replace the value below with those
    # sentence labels.
    sent_labels = test_sents

    with torch.no_grad():
        for sent in sent_labels:
            if len(decX[sent]) > 1:

                values = m.analyze_sentence(encX[sent], decX[sent], memory_length=len(decX))
                hidden_values = np.array([sent]+values[-1].view(-1).detach().numpy().tolist()).reshape(1, -1)
                hidden_df = pd.DataFrame(hidden_values, columns=list(hidden_df))
                hidden_df.to_csv(states_out, header=False, index=False, mode='a', encoding='utf-8')

                if outputs_out:
                    #outout = np.array([sent, outputs.topk(1, dim=-1)[1].view(-1)[-1].item()]).reshape(1, -1)
                    outout = np.array([sent] + values[0].topk(3, dim=-1)[1][0][-1].tolist()).reshape(1, -1)
                    outputs_df = pd.DataFrame(outout, columns=list(outputs_df))
                    outputs_df.to_csv(outputs_out, header=False, index=False, mode='a', encoding='utf-8')

                if attn_data:
                    lexemes = attn_data[1]['lex'].loc[attn_data[1]['sent'].isin([sent])].values
                    cm = dec_all['cmsource'].loc[dec_all['sent'].isin([sent])].unique()[0]

                    dec_steps = dec_all[attn_data[2]].loc[dec_all['sent'].isin([sent])].values[0].reshape(-1)

                    for i, step in enumerate(values[1]):
                        k = step.detach().numpy().reshape(-1)
                        topN = str(lexemes[k.argsort()[-1:-6:-1]].tolist()).replace('\'', '').replace(', ','\t').replace('[', '').replace(']','') #' '.join(lexemes[k.argsort()[-1:-4:-1]])
                        alpha_df = pd.DataFrame(np.array([sent, cm, dec_steps[i],
                                                          str(k.tolist()).replace('[', '').replace(']',''),
                                                          topN]).reshape(-1, 5), columns=list(alpha_df))
                        alpha_df.to_csv(attn_data[0], header=False, index=False, mode='a', encoding='utf-8')
                #print(sent)

    print('data structure built')

    hidden_df = pd.read_csv(states_out)
    for col in state_relational_columns:
        try:
            hidden_df[col] = [dec_all[col].loc[dec_all['sent'].isin([sent])].unique()[0]
                              for sent in hidden_df['sent'].values]
        except KeyError:
            print(col, ' not in data')
    hidden_df.to_csv(states_out, header=True, index=False, encoding='utf-8')
    print('data augmented with relevant features')

    return alpha_df, hidden_df

def cSave(m, encX, decX, test_sents, dec_all, states_out, state_relational_columns=['sentID', 'tref.stem', 'head.stem', 'tref.dep', 'cmsource', 'span'], attn_data=None, outputs_out=None):
    """

    :param m: the model being used to generate our interpretations.
    :param encX: encoder data
    :param decX: decoder data
    :param test_sents: the sentence labels for the sentences that we're going to be testing on
    :param enc_all: the originally imported document for our encoder data. Even post conversion, it still holds a lot of really useful info.
    :param dec_all: the originally imported document for our decoder data. Even post conversion, it still holds a lot of really useful info.
    :param attn_data: a list or tuple containing at index0 an output file location, index1 the encoder data used to find qualitative weights, and optional index2 a list of cols used for steps.
    :param horizontal: Occasionally, our values for decoder steps are going to be horizontal, per sentence. This triggers that.
    :param streaming: must include 'encoder_df': encoder_df, 'decoder_fx': decoder_df, 'encoder_fx': (function, cols), 'decoder_fx': (function, cols)
    :return:
    """
    m.enc.eval()
    m.dec.eval()
    m.gATTN.eval()
    #m.out.eval()

    alpha_df = None
    if attn_data:
        alpha_df = pd.DataFrame(columns=['sent', 'cm', 'topic', 'head', 'step', 'weights', 'topN'])
        alpha_df.to_csv(attn_data[0], index=False, encoding='utf-8')

    hidden_df = pd.DataFrame(columns=['sent', 'state'])
    hidden_df.to_csv(states_out, index=False, encoding='utf-8')

    outputs_df = None
    if outputs_out:
        outputs_df = pd.DataFrame(columns=['sent', 'tag1', 'tag2', 'tag3']) #else columns=['sent', 'tag']
        outputs_df.to_csv(outputs_out, index=False, encoding='utf-8')

    #If for some reason you have some subset of sentences in the
    # data that you wish to run testing procedures on, besides
    # the cummulative data, replace the value below with those
    # sentence labels.
    sent_labels = test_sents

    with torch.no_grad():
        for sent in sent_labels:
            if len(decX[sent]) > 1:

                values = m.analyze_sentence(encX[sent], decX[sent], memory_length=len(decX))
                hidden_values = np.array([sent, str(values[-1].detach()).replace('\n', 'NEWLINE')]).reshape(1, -1)
                hidden_df = pd.DataFrame(hidden_values, columns=list(hidden_df))
                hidden_df.to_csv(states_out, header=False, index=False, mode='a', encoding='utf-8')

                if outputs_out:
                    #outout = np.array([sent, outputs.topk(1, dim=-1)[1].view(-1)[-1].item()]).reshape(1, -1)
                    outout = np.array([sent] + values[0].topk(3, dim=-1)[1][0][-1].tolist()).reshape(1, -1)
                    outputs_df = pd.DataFrame(outout, columns=list(outputs_df))
                    outputs_df.to_csv(outputs_out, header=False, index=False, mode='a', encoding='utf-8')

                if attn_data:
                    lexemes = attn_data[1]['lex'].loc[attn_data[1]['sent'].isin([sent])].values
                    cm = dec_all['cmsource'].loc[dec_all['sent'].isin([sent])].unique()[0]
                    topic = dec_all['tref.stem'].loc[dec_all['sent'].isin([sent])].unique()[0]
                    head = dec_all['head.stem'].loc[dec_all['sent'].isin([sent])].unique()[0]

                    dec_steps = dec_all[attn_data[2]].loc[dec_all['sent'].isin([sent])].values

                    for i, step in enumerate(values[1]):
                        k = step.detach().numpy().reshape(-1)
                        topN = str(lexemes[k.argsort()[-1:-6:-1]].tolist()).replace('\'', '').replace(', ','\t').replace('[', '').replace(']','') #' '.join(lexemes[k.argsort()[-1:-4:-1]])
                        alpha_df = pd.DataFrame(np.array([sent, cm, topic, head, dec_steps[i],
                                                          str(k.tolist()).replace('[', '').replace(']',''),
                                                          topN]).reshape(-1, 7), columns=list(alpha_df))
                        alpha_df.to_csv(attn_data[0], header=False, index=False, mode='a', encoding='utf-8')
                #print(sent)

    print('data structure built')

    hidden_df = pd.read_csv(states_out)
    for col in state_relational_columns:
        try:
            hidden_df[col] = [dec_all[col].loc[dec_all['sent'].isin([sent])].unique()[0]
                              for sent in hidden_df['sent'].values]
        except KeyError:
            print(col, ' not in data')
    hidden_df.to_csv(states_out, header=True, index=False, encoding='utf-8')
    print('data augmented with relevant features')

    return alpha_df, hidden_df


def cSave__(m, encX, decX, test_sents, dec_all, alpha_out, states_out, indX, statesize=1200, state_relational_columns=['sentID', 'tref.stem', 'head.stem', 'tref.dep', 'cmsource', 'span']):
    """

    :param m: the model being used to generate our interpretations.
    :param encX: encoder data
    :param decX: decoder data
    :param test_sents: the sentence labels for the sentences that we're going to be testing on
    :param enc_all: the originally imported document for our encoder data. Even post conversion, it still holds a lot of really useful info.
    :param dec_all: the originally imported document for our decoder data. Even post conversion, it still holds a lot of really useful info.
    :param horizontal: Occasionally, our values for decoder steps are going to be horizontal, per sentence. This triggers that.
    :param streaming: must include 'encoder_df': encoder_df, 'decoder_fx': decoder_df, 'encoder_fx': (function, cols), 'decoder_fx': (function, cols)
    :return:
    """
    m.enc.eval()
    m.dec.eval()
    m.gATTN.eval()
    m.out.eval()

    alpha_df = pd.DataFrame(columns=['sent', 'lex', 'weights', 'enc.states'])
    alpha_df.to_csv(alpha_out, index=False, encoding='utf-8')

    hidden_df = pd.DataFrame(columns=['sent']+[str(i) for i in range(statesize)])
    hidden_df.to_csv(states_out, index=False, encoding='utf-8')

    #If for some reason you have some subset of sentences in the
    # data that you wish to run testing procedures on, besides
    # the cummulative data, replace the value below with those
    # sentence labels.
    sent_labels = test_sents

    with torch.no_grad():
        for sent in sent_labels:
            if len(decX[sent]) > 1:
                indxs = indX[sent]
                _, alphas, _, _, hiddens = m.analyze_sentence(encX[sent], decX[sent], memory_length=len(decX))

                hidden_values = np.array([sent]+hiddens.squeeze(0)[indxs[1:]].view(-1).detach().numpy().tolist()).reshape(1, -1)
                hidden_df = pd.DataFrame(hidden_values, columns=list(hidden_df))
                hidden_df.to_csv(states_out, header=False, index=False, mode='a', encoding='utf-8')

    print('data structure built')

    hidden_df = pd.read_csv(states_out)
    for col in state_relational_columns:
        hidden_df[col] = [dec_all[col].loc[dec_all['sent'].isin([sent])].unique()[0]
                          for sent in hidden_df['sent'].values]
    hidden_df.to_csv(states_out, header=True, index=False, encoding='utf-8')
    print('data augmented with relevant features')

    return alpha_df, hidden_df
