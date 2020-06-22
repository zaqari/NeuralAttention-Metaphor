import torch.nn as nn
import torch.nn.functional as F
import torch
from random import randint
import random
import numpy as np


class model(nn.Module):

    def __init__(self, encoder, decoder, global_attention, memory_size=(1000, 2, 600), noise_step=1e-5):
        """

        :param encoder:
        :param decoder:
        :param outlayer:
        :param streaming: must include 'df': df, 'encoder_data': cols, 'decoder_data': cols
        """
        super(model, self).__init__()
        self.enc = encoder
        self.dec = decoder
        self.gATTN = global_attention
        self.one_hot_random_ = torch.eye(self.dec.n_classes)

        ##### Memory functions
        # Must be defined manually
        # One way to do this for a variable length memory stack, could look like
        #  the following, external to the class
        #   stop = int(len(df['sent'].unique()/3)
        #   a, b, c = df['sent'].unique()[:stop], df['sent'].unique()[stop:stop * 2], df['sent'].unique()[stop * 2:len(df['sent'].unique())]
        #   sent_dic = sum([[(i, list(it).index(i)) for i in it] for it in [a,b,c]], [])
        #   sent_dic = {i[0]:i[1] for i in sent_dic}
        # This is fairly efficient overall, though not perfectly so.
        self.softsign = nn.Softsign()  # We'll use this to artificially add noise to our training of the cosBahdanau function
        self.pull_memories = False
        self.noise_step = noise_step
        self.memory = torch.zeros(size=memory_size, requires_grad=False)

    def analyze_sentence(self, encoder_data, decoder_data, memory_length=100):
        """
        Full model utilization for forward. In this
         instance, mode=0 is .train() mode, mode=1 will be
         .eval() mode.
        For the moment, we're giving this a shot as the
         generator for the training period, too, though if
         need be we can switch it out to run the generator
         in line again.
        """
        ##### Typical Encoder-Decoder steps
        encoder_outputs, hiddn = self.encode(encoder_data)
        hidden = torch.cat([hiddn, nn.init.xavier_uniform_(self.dec.initHidden())], dim=0)
        projected_keys = self.dec.attention.proj_layer(encoder_outputs)
        outputs, pre_outputs, alphas, hidden = self.decode(decoder_data, hidden, projected_keys, encoder_outputs)

        ##### Global memory
        #global_outputs, global_context = self.remember(pre_outputs, sample_n=memory_length)

        return outputs, alphas, pre_outputs, pre_outputs

    def encode(self, encoder_data):
        encoder_outputs = []
        hidden = self.enc.initHidden()
        for word in range(len(encoder_data)):
            output, hidden = self.enc(encoder_data[word].view(-1), hidden)
            encoder_outputs.append(output)
        encoder_outputs = torch.cat(encoder_outputs, dim=1).squeeze(0)
        return encoder_outputs, hidden

    def remember(self, local_pre_outputs, sample_n=50):
        if self.pull_memories:
            #sample_order = torch.tensor(np.random.choice(len(self.memory), size=(sample_n,), replace=False))
            #return self.gATTN(local_pre_outputs, self.gATTN.relu(self.gATTN.projection(self.memory[sample_order])))
            return local_pre_outputs, local_pre_outputs
        else:
            return local_pre_outputs, local_pre_outputs

    def decode(self, decoder_inputs, encoder_hidden_state, projected_keys, encoder_outputs):
        outputs, pre_outputs, alphas, hidden = self.dec(decoder_inputs, encoder_hidden_state, projected_keys,
                                                        encoder_outputs)
        return outputs, pre_outputs, alphas, hidden

    def batched_training(self, enc_x, dec_x, y, sentences, optimizer, loss_criterion, validation_data=(), epochs=10,
                          cutoff=.9, sample_pct=1.0, pull_from_memory=False, memory_size=50, noise_step=False):

        self.pull_memories = pull_from_memory
        if noise_step:
            self.noise_step = noise_step
        """ """
        steps = int(len(sentences) / len(self.memory))
        loci = sum([list(range(0, len(self.memory))) for _ in range(steps)], [])
        if len(loci) < len(sentences):
            loci += list(range(0, len(sentences) - len(loci)))
        if len(loci) > len(sentences):
            loci = loci[:len(sentences)]

        sent2id = {sent: loci[sentences.index(sent)] for sent in sentences}

        for epoch in range(epochs):

            self.enc.train()
            self.dec.train()
            self.gATTN.train()
            #self.out.train()
            y.train()
            optimizer.zero_grad()

            print('Epoch {}/{}'.format(epoch + 1, epochs))

            samples = np.random.choice(sentences, int(len(sentences) * sample_pct), replace=False).tolist() #[sentences[i] for i in list(np.random.choice(len(enc_x), int(len(enc_x) * sample_pct), replace=False).reshape(-1))]
            for sent in samples:
                #self.memory[sent2id[sent]] = torch.zeros(size=(self.memory.shape[1:]))

                outputs, _, pre_outputs, global_outputs = self.analyze_sentence(enc_x[sent], dec_x[sent], memory_length=memory_size)
                #memory_update = pre_outputs.detach()

                #loss = loss_criterion(outputs.view(1, -1, self.dec.n_classes), y[sent].view(1, -1, self.dec.n_classes))

                #IF USING CosineEmbeddingLoss
                loss = loss_criterion(outputs, y[sent].view(outputs.shape), torch.ones(size=outputs.shape))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                #self.memory[sent2id[sent]] = memory_update

            cut = self.epoch_statistics(validation_data=validation_data, train_data=(enc_x, dec_x, y, sentences),
                                         loss_criterion=loss_criterion)
            if cut <= cutoff:
                break

    def epoch_statistics(self, validation_data, train_data, loss_criterion):
        cut_off = None
        if bool(validation_data):
            cut_off = self.evaluation(validation_data[0], validation_data[1], validation_data[2], validation_data[3],
                                       loss_criterion)
        else:
            cut_off = self.evaluation(train_data[0], train_data[1], train_data[2], train_data[3], loss_criterion)
        return cut_off

    def evaluation(self, x, dec_x, Y, sentences, loss_criterion):

        self.enc.eval()
        self.dec.eval()
        self.gATTN.eval()
        #self.out.eval()
        Y.eval()

        memoryN = len(self.memory)
        with torch.no_grad():
            accuracy, lossiness = [], []
            for i in sentences:
                outputs, _, _, _ = self.analyze_sentence(x[i], dec_x[i], memory_length=memoryN)
                #lossiness += [loss_criterion(outputs.view(1, -1, self.dec.n_classes), Y[i].view(1, -1, self.dec.n_classes))]
                #IF USING COS_SIM LOSS
                lossiness += [loss_criterion(outputs, Y[i].view(outputs.shape), torch.ones(size=outputs.shape))]
            print('@los: {:.4f}'.format(sum(lossiness) / len(lossiness)))
            print('============] [============\n')
            # self.gATTN.novelty_bias = prior_bias
            return sum(lossiness) / len(lossiness)
