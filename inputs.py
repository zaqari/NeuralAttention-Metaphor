import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer, LancasterStemmer, SnowballStemmer
import torch

####################################################################################
######## Padding trigger
####################################################################################
#Set to None if no padding required.
nPad = None


####################################################################################
######## Lemmatization
####################################################################################
lem = SnowballStemmer('english')
nonce = 'kelilili'

print("WARNING! cData is temporarily changed to only continue on to [:-1] in the sequence!")

class rData(torch.utils.data.Dataset):

    def __init__(self, df, columns, conversion_dic, permissable_sentences, oncol='sent', stemming=True, SOS_token=False):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]
        self.ex_col = oncol
        self.converter = conversion_dic
        self.columns = columns
        self.stem = stemming
        self.SOS = SOS_token

    def __len__(self):
        return len(self.df[self.ex_col].unique())

    def __getitem__(self, sent):
        data = self.df[self.columns].loc[self.df[self.ex_col].isin([sent])]
        data = data.loc[data.index[0]].values.tolist()
        if self.stem:
            data = [lem.stem(str(w)) for w in data]
        data = [self.converter[w] for w in data]
        if self.SOS:
            data.insert(0, 0)
        return torch.LongTensor(data).view(-1, 1)

class cData(torch.utils.data.Dataset):

    def __init__(self, df, column, conversion_dic, permissable_sentences, oncol='sent', stemming=True, SOS_token=False):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]
        self.ex_col = oncol
        self.converter = conversion_dic
        self.col = column
        self.stem = stemming
        self.SOS = SOS_token

    def __len__(self):
        return len(self.df[self.ex_col].unique())

    def __getitem__(self, sent):
        data = self.df[self.col].loc[self.df[self.ex_col].isin([sent])].values.tolist()
        if self.stem:
            data = [lem.stem(str(w)) for w in data]

        data = [self.converter[w] for w in data]

        if self.SOS:
            #data[0] = 0
            data.insert(0,0)

        return torch.LongTensor(data).view(-1, 1)


class cDataALT(torch.utils.data.Dataset):

    def __init__(self, df, column, conversion_dic, permissable_sentences, oncol='sent', stemming=True, SOS_token=False):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]
        self.ex_col = oncol
        self.converter = conversion_dic
        self.col = column
        self.stem = stemming
        self.SOS = SOS_token

    def __len__(self):
        return len(self.df[self.ex_col].unique())

    def __getitem__(self, sent):
        local = self.df.loc[self.df[self.ex_col].isin([sent])]
        data = local[self.col].values.tolist()
        data.insert(0, local['tref'].values[0])
        if self.stem:
            data = [lem.stem(str(w)) for w in data]

        data = [self.converter[w] for w in data]

        if self.SOS:
            data[0] = 0

        return torch.LongTensor(data).view(-1, 1)

class cIndx(torch.utils.data.Dataset):
    def __init__(self, df, colsets, permissable_sentences, oncol='sent'):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]
        self.ex_col = oncol
        self.colsets = colsets

    def __len__(self):
        return len(self.df[self.ex_col].unique())

    def __getitem__(self, sent):
        data = self.df.loc[self.df[self.ex_col].isin([sent])]
        out = [(data[col[0]] == data[col[1]]).values for col in self.colsets]
        out = np.sum(out, axis=0, dtype=bool)
        out[0] = False
        return out