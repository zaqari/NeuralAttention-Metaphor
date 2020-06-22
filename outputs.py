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

class rdVecs(torch.utils.data.Dataset):

    def __init__(self, df, columns, conversion_dic, permissable_sentences, vector_model, oncol='sent', stemming=True, SOS_token=False):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]
        self.ex_col = oncol
        self.converter = conversion_dic
        self.columns = columns
        self.stem = stemming
        self.SOS = SOS_token
        self.vecs = vector_model
        self.vec_size = self.converter[lem.stem(str(self.df[columns[0]].loc[list(self.df.index)[0]]))]
        self.vec_size = self.vecs[self.vec_size].shape[-1]

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
        data = [self.vecs[data[i + 1]] - self.vecs[data[i]] for i in range(len(data)-1)]
        return torch.FloatTensor(data).view(-1, self.vec_size)

class cdVecs(torch.utils.data.Dataset):

    def __init__(self, df, column, conversion_dic, permissable_sentences, vector_model, oncol='sent', stemming=True, SOS_token=False):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]
        self.ex_col = oncol
        self.converter = conversion_dic
        self.col = column
        self.stem = stemming
        self.SOS = SOS_token

        self.vecs = vector_model
        self.vec_size = self.converter[lem.stem(str(self.df[column].loc[list(self.df.index)[0]]))]
        self.vec_size = self.vecs[self.vec_size].shape[-1]

    def __len__(self):
        return len(self.df[self.ex_col].unique())

    def __getitem__(self, sent):
        data = self.df[self.col].loc[self.df[self.ex_col].isin([sent])].values.tolist()
        if self.stem:
            data = [lem.stem(str(w)) for w in data]
        data = [self.converter[w] for w in data]
        if self.SOS:
            data[0] = 0
        data = [self.vecs[data[i + 1]] - self.vecs[data[i]] for i in range(len(data)-1)]
        return torch.FloatTensor(data).view(-1, self.vec_size)


class rVecs(torch.utils.data.Dataset):

    def __init__(self, df, columns, conversion_dic, permissable_sentences, vector_model, oncol='sent', stemming=True, SOS_token=False):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]
        self.ex_col = oncol
        self.converter = conversion_dic
        self.columns = columns
        self.stem = stemming
        self.SOS = SOS_token
        self.vecs = vector_model
        self.vec_size = self.converter[lem.stem(str(self.df[columns[0]].loc[list(self.df.index)[0]]))]
        self.vec_size = self.vecs[self.vec_size].shape[-1]
        self.training = True

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
        data = [self.vecs[i] for i in data[1:]]
        return torch.FloatTensor(data).view(-1, self.vec_size)

    def train(self):
        self.training = True

    def eval(self):
        self.training=False

class cVecs(torch.utils.data.Dataset):

    def __init__(self, df, column, conversion_dic, permissable_sentences, vector_model, oncol='sent', stemming=True, SOS_token=False):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]
        self.ex_col = oncol
        self.converter = conversion_dic
        self.col = column
        self.stem = stemming
        self.SOS = SOS_token

        self.training = True

        self.vecs = vector_model
        convert_1 = lem.stem(self.df[column].loc[list(self.df.index)[0]].values[0])

        self.vec_size = self.converter[convert_1]
        self.vec_size = self.vecs[self.vec_size].shape[-1]

    def __len__(self):
        return len(self.df[self.ex_col].unique())

    def __getitem__(self, sent):
        data = self.df[self.col].loc[self.df[self.ex_col].isin([sent])].values.reshape(-1).tolist()
        if self.stem:
            data = [lem.stem(str(w)) for w in data]
        data = [self.converter[w] for w in data]
        if self.SOS:
            #data[0] = 0
            data.insert(0,0)
        data = [self.vecs[i] for i in data[1:]]

        return torch.FloatTensor(data).view(-1, self.vec_size)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class rEVecs(torch.utils.data.Dataset):

    def __init__(self, df, columns, conversion_dic, permissable_sentences, vector_model, oncol='sent', stemming=True, SOS_token=False):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]
        self.ex_col = oncol
        self.converter = conversion_dic
        self.columns = columns
        self.stem = stemming
        self.SOS = SOS_token
        self.vecs = vector_model
        self.vec_size = self.converter[lem.stem(str(self.df[columns[0]].loc[list(self.df.index)[0]]))]
        self.vec_size = self.vecs[self.vec_size].shape[-1]

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
        data = [self.vecs[data[-1]] - self.vecs[data[i]] for i in range(len(data)-1)]
        return torch.FloatTensor(data).view(-1, self.vec_size)

class cEVecs(torch.utils.data.Dataset):

    def __init__(self, df, column, conversion_dic, permissable_sentences, vector_model, oncol='sent', stemming=True, SOS_token=False):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]
        self.ex_col = oncol
        self.converter = conversion_dic
        self.col = column
        self.stem = stemming
        self.SOS = SOS_token

        self.vecs = vector_model
        self.vec_size = self.converter[lem.stem(str(self.df[column].loc[list(self.df.index)[0]]))]
        self.vec_size = self.vecs[self.vec_size].shape[-1]

    def __len__(self):
        return len(self.df[self.ex_col].unique())

    def __getitem__(self, sent):
        data = self.df[self.col].loc[self.df[self.ex_col].isin([sent])].values.tolist()
        if self.stem:
            data = [lem.stem(str(w)) for w in data]
        data = [self.converter[w] for w in data]
        if self.SOS:
            data[0] = 0
        data = [self.vecs[data[-1]] - self.vecs[data[i]] for i in range(len(data)-1)]
        return torch.FloatTensor(data).view(-1, self.vec_size)


class rRLVecs(torch.utils.data.Dataset):

    def __init__(self, df, columns, conversion_dic, permissable_sentences, vector_model, train_sentences, oncol='sent', stemming=True, SOS_token=False, epsilon=.3, degradation=.001):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]

        self.ex_col = oncol
        self.converter = conversion_dic
        self.columns = columns
        self.stem = stemming
        self.SOS = SOS_token

        self.vecs = vector_model
        self.vec_size = self.converter[lem.stem(str(self.df[columns[0]].loc[list(self.df.index)[0]]))]
        self.vec_size = self.vecs[self.vec_size].shape[-1]

        self.training = True
        self.train_sentences = train_sentences
        self.eps = epsilon
        self.eps_degradation = degradation

    def __len__(self):
        return len(self.df[self.ex_col].unique())

    def __getitem__(self, sent):
        data = self.df.loc[self.df[self.ex_col].isin([sent])]
        data = data[self.columns].loc[data.index[0]].values.tolist()

        if self.training:

            if np.random.rand() > self.eps:
                idx = np.random.choice(self.df.index[self.df['cmsource.1'].isin(self.df['cmsource.1'].loc[self.df['sent'].isin([sent])].values) &
                                                     self.df['sent'].isin(self.train_sentences)].values,
                                       size=1, replace=False)[0]
                data[1:] = self.df[self.columns[1:]].loc[idx].values.tolist()

        if self.stem:
            data = [lem.stem(str(w)) for w in data]

        data = [self.converter[w] for w in data]

        if self.SOS:
            data.insert(0, 0)

        data = [self.vecs[data[i+1]] - self.vecs[data[i]] for i in range(len(data)-1)]
        
        return torch.FloatTensor(data).view(-1, self.vec_size)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class rRLVecs2(torch.utils.data.Dataset):

    def __init__(self, df, columns, conversion_dic, permissable_sentences, vector_model, train_sentences, oncol='sent',
                 stemming=True, SOS_token=False, epsilon=.3, degradation=.001):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]

        self.ex_col = oncol
        self.converter = conversion_dic
        self.columns = columns
        self.stem = stemming
        self.SOS = SOS_token

        self.vecs = vector_model
        self.vec_size = self.converter[lem.stem(str(self.df[columns[0]].loc[list(self.df.index)[0]]))]
        self.vec_size = self.vecs[self.vec_size].shape[-1]

        self.training = True
        self.train_sentences = train_sentences
        self.eps = epsilon
        self.eps_degradation = degradation

    def __len__(self):
        return len(self.df[self.ex_col].unique())

    def __getitem__(self, sent):
        data = self.df.loc[self.df[self.ex_col].isin([sent])]
        data = data[self.columns].loc[data.index[0]].values.tolist()

        if self.training:

            if np.random.rand() > self.eps:

                idx = np.random.choice(self.df.index[self.df['cmsource'].isin(
                    self.df['cmsource'].loc[self.df['sent'].isin([sent])].values) &
                                                     self.df['sent'].isin(self.train_sentences)].values,
                                       size=len(self.columns[1:]))

                data[1:] = [str(self.df[self.columns[i+1]].loc[idx[0]]) for i in range(len(idx))]

        if self.stem:
            data = [lem.stem(str(w)) for w in data]

        data = [self.converter[w] for w in data]

        if self.SOS:
            data.insert(0, 0)

        #BEST   data = [self.vecs[data[i + 1]] - self.vecs[data[i]] for i in range(len(data) - 1)]
        data = [self.vecs[i] for i in data[1:]]

        return torch.FloatTensor(data).view(-1, self.vec_size)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

        
class rRLVecs3(torch.utils.data.Dataset):

    def __init__(self, df, columns, conversion_dic, permissable_sentences, vector_model, train_sentences, oncol='sent',
                 stemming=True, SOS_token=False, epsilon=.3, degradation=.001):
        self.df = df.loc[df[oncol].isin(permissable_sentences)]

        self.ex_col = oncol
        self.converter = conversion_dic
        self.columns = columns
        self.stem = stemming
        self.SOS = SOS_token

        self.vecs = vector_model
        self.vec_size = self.converter[lem.stem(str(self.df[columns[0]].loc[list(self.df.index)[0]]))]
        self.vec_size = self.vecs[self.vec_size].shape[-1]

        self.training = True
        self.train_sentences = train_sentences
        self.eps = epsilon
        self.eps_degradation = degradation

    def __len__(self):
        return len(self.df[self.ex_col].unique())

    def __getitem__(self, sent):
        data = self.df.loc[self.df[self.ex_col].isin([sent])]
        data = data[self.columns].loc[data.index[0]].values.tolist()

        if self.training:

            if np.random.rand() > self.eps:

                idx = np.random.choice(self.df.index[self.df['cmsource'].isin(
                    self.df['cmsource'].loc[self.df['sent'].isin([sent])].values) &
                                                     self.df['sent'].isin(self.train_sentences)].values,
                                       size=len(self.columns[:2]))

                data[:len(self.columns)-1] = [str(self.df[self.columns[i]].loc[idx[0]]) for i in range(len(idx))]

        if self.stem:
            data = [lem.stem(str(w)) for w in data]

        data = [self.converter[w] for w in data]

        if self.SOS:
            data.insert(0, 0)

        #BEST   data = [self.vecs[data[i + 1]] - self.vecs[data[i]] for i in range(len(data) - 1)]
        data = [self.vecs[i] for i in data[1:]]

        return torch.FloatTensor(data).view(-1, self.vec_size)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

