import torch
import torch.nn as nn
import torch.nn.functional as F

class enc(nn.Module):

    def __init__(self, device, embeddings=[], input_shape=1, hidden_size=300, bidirectional=True, dropout=.2):
        super(enc, self).__init__()

        self.device = device
        self.bi = bidirectional
        self.hidden_size = hidden_size

        self.output_size = hidden_size
        if self.bi:
            self.output_size *= 2

        #Embedding layer
        embeds = torch.FloatTensor(embeddings)
        self.embedding = nn.Embedding.from_pretrained(embeds)
        self.embedding.requires_grad = True

        #Encoder net
        self.encoder_gru = nn.GRU(self.embedding.embedding_dim,
                                  self.hidden_size,
                                  bidirectional=self.bi)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        output, hidden = self.encoder_gru(output, hidden)
        output = self.relu(output)
        output = self.drop(output)
        return output, hidden

    def initHidden(self):
        if self.bi:
            return torch.zeros(1, 1, self.hidden_size, device=self.device)
        else:
            return torch.zeros(1, 1, self.hidden_size, device=self.device)