import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class dec_SAVE(nn.Module):

    def __init__(self, device, attention, embeddings, outputlayer, hidden_size=300, context_size=600, dropout_p=.1, bidirectional=True, SOS_token=False):
        super(dec_SAVE, self).__init__()

        self.cos = nn.CosineSimilarity(dim=-1)
        self.device = device
        self.n_layers = 2
        self.hidden_size = hidden_size
        self.n_classes = len(embeddings) + 1
        self.bi = bidirectional
        self.output_size = self.hidden_size

        self.SOS = SOS_token

        self.bi_mul = 1
        if self.bi:
            self.bi_mul = 2

        #Embedding layer
        self.embeds = torch.FloatTensor(embeddings)
        #embed_size = self.embedding.shape[-1]
        #self.embedding = nn.Embedding.from_pretrained(self.embeds)
        #self.embedding = nn.Embedding(len(embeddings), len(embeddings[0]))
        self.embed_size = len(embeddings[0])
        #self.embeds = embeddings

        #Layers
        self.attention = attention
        self.outlayer = outputlayer
        self.dropout = nn.Dropout(p=dropout_p)
        self.gru1 = nn.GRU(self.embed_size + context_size,
                           self.hidden_size,
                           bidirectional=self.bi)
        self.gru2 = nn.GRU(self.hidden_size * self.bi_mul + context_size,
                           self.hidden_size,
                           bidirectional=self.bi)
        self.pre_output_layer = nn.Linear(hidden_size + context_size + self.embed_size, hidden_size)
        self.relu = nn.LeakyReLU()

    def lexical_step(self, input, hidden, projected_keys, encoder_outputs):
        """
        The following is a step-wise function that takes some input,
        the projected versions of the encoder states (states are reduced
        to hidden_size-D using the projection layer prima facie), the
        hidden state from the prior step, and the actual encoder_outputs
        to create a representation of the word going into the decoder
        and its context at t=sentence.index(word)
        """
        hidden_states = []

        #(1) Convert input to an embedding.
        embedded = self.embeds[input.item()].view(1,1,-1)
        #(2) Generate the context vector and alphas using the attention
        #    layer provided to the network on initialization.
        context, alphas = self.attention(hidden, projected_keys, encoder_outputs)

        # (3) GRU-1 Ops
        # (3.1) Concatenate the embedding rep of the word with the
        #       generated context vector.
        gru1_input = torch.cat([embedded, context], dim=-1)
        # (3.2) Pass the previous concatenation as input to the
        #       GRU-1 layer
        output, hidN = self.gru1(gru1_input, hidden[0])
        # (3.3) Apply relu function and dropout to output, in order to
        #       avoid disappearing gradients and overfiting,
        #       respectively.
        output = self.dropout(self.relu(output))
        # (3.4) Append hidden state to the hidden states index.
        hidden_states.append(hidN)

        # (4) GRU-2 Ops:
        #     GRU-2 is unique in comparison to -1, but the operations
        #     are mirrored.
        gru2_input = torch.cat([output, context], dim=-1)
        output, hidN = self.gru2(gru2_input, hidden[-1])
        output = self.dropout(self.relu(output))
        hidden_states.append(hidN)

        # (5) Output Manipulation
        # (5.1) Concatenate embedding, output, and context vectors
        #       to be passed to the softmax, output layer. This gives
        #       our output access to all information pertaining to
        #       this time-step.
        pre_output = torch.cat([embedded, output, context], dim=-1)
        # (5.2) Optionally, apply dropout to (5.1).
        pre_output = self.dropout(pre_output)
        # (5.3) Now, compress this information to hidden_size, and
        #       simultaneously allow for a grad and differentiation to
        #       be applied here!
        pre_output = self.relu(self.pre_output_layer(pre_output))

        return output, hidden_states, pre_output, alphas

    def forward(self, inputs, encoder_hidden_state, projected_keys, encoder_outputs, teacher_forcing=.4):
        """
        The following takes in the entirety of the inputs for the
        decoder, passes it to our stepwise function, lexical_step(),
        and iterates through the sentence to create a useful output.
        """
        hidden, outputs, alphas_out, pre_outputs = encoder_hidden_state, [], [], []

        # 10/26 Changed loss such that true_y starts at n+1 ... this
        # was done in order to allow the model to start at decoder
        # inputs[0].
        xi = torch.tensor(inputs[0].item())
        for i in range(len(inputs)-1):
            output, hidden, pre_output, alphas = self.lexical_step(xi, hidden, projected_keys, encoder_outputs)
            output = self.outlayer(output)

            outputs.append(output)
            pre_outputs.append(pre_output)
            alphas_out.append(alphas)

            #implements a probabilistic teacher forcing shedule
            if np.random.rand() >= teacher_forcing:
                xi = torch.tensor(self.cos(output.view(1,-1), self.embeds).topk(k=1, dim=-1)[1].item())

            else:
                xi = torch.tensor(inputs[i+1].item())

        return torch.cat(outputs, dim=1), torch.cat(pre_outputs, dim=1), alphas_out, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class dec_2(nn.Module):

    def __init__(self, device, attention, embeddings, outputlayer, hidden_size=300, context_size=600, dropout_p=.1, bidirectional=True, SOS_token=False):
        super(dec_2, self).__init__()

        self.cos = nn.CosineSimilarity(dim=-1)
        self.device = device
        self.n_layers = 2
        self.hidden_size = hidden_size
        self.n_classes = len(embeddings) + 1
        self.bi = bidirectional
        self.output_size = self.hidden_size

        self.SOS = SOS_token

        self.bi_mul = 1
        if self.bi:
            self.bi_mul = 2

        #Embedding layer
        self.embeds = torch.FloatTensor(embeddings)
        #embed_size = self.embedding.shape[-1]
        #self.embedding = nn.Embedding.from_pretrained(self.embeds)
        #self.embedding = nn.Embedding(len(embeddings), len(embeddings[0]))
        self.embed_size = len(embeddings[0])
        #self.embeds = embeddings

        #Layers
        self.attention = attention
        self.outlayer = outputlayer
        self.dropout = nn.Dropout(p=dropout_p)
        self.gru = nn.GRU(self.embed_size + context_size,
                           self.hidden_size,
                           bidirectional=self.bi)

        self.gru2 = nn.GRU(self.hidden_size,
                          self.hidden_size,
                          bidirectional=self.bi)

        self.pre_output_layer = nn.Linear(self.hidden_size + context_size, self.hidden_size)
        self.relu = nn.LeakyReLU()

    def lexical_step(self, input, hidden, projected_keys, encoder_outputs):
        """
        The following is a step-wise function that takes some input,
        the projected versions of the encoder states (states are reduced
        to hidden_size-D using the projection layer prima facie), the
        hidden state from the prior step, and the actual encoder_outputs
        to create a representation of the word going into the decoder
        and its context at t=sentence.index(word)
        """
        hidden_states = []

        #(1) Convert input to an embedding.
        embedded = self.embeds[input.item()].view(1,1,-1)
        embedded = self.dropout(embedded)

        # (2) Generate the context vector and alphas using the attention
        #    layer provided to the network on initialization.
        context, alphas = self.attention(hidden[0], projected_keys, encoder_outputs)
        gru1_in = torch.cat([embedded, context], dim=-1)

        # (3) GRU-1 Ops
        # (3.1) Concatenate the embedding rep of the word with the
        #       generated context vector.
        #gru1_input = torch.cat([embedded, context], dim=-1)
        # (3.2) Pass the previous concatenation as input to the
        #       GRU-1 layer
        output, hidN = self.gru(gru1_in, hidden[0])
        hidden_states.append(hidN)

        output = self.dropout(self.relu(output))

        output, hidN = self.gru2(output, hidden[-1])
        hidden_states.append(hidN)

        # (3.3) Apply relu function and dropout to output, in order to
        #       avoid disappearing gradients and overfiting,
        #       respectively.
        output = self.dropout(self.relu(output))

        # (5) Output Manipulation
        # (5.1) Concatenate embedding, output, and context vectors
        #       to be passed to the softmax, output layer. This gives
        #       our output access to all information pertaining to
        #       this time-step.
        pre_output = torch.cat([output, context], dim=-1)
        # (5.2) Optionally, apply dropout to (5.1).
        pre_output = self.dropout(pre_output)
        # (5.3) Now, compress this information to hidden_size, and
        #       simultaneously allow for a grad and differentiation to
        #       be applied here!
        pre_output = self.relu(self.pre_output_layer(pre_output))

        return output, hidden_states, pre_output, alphas

    def forward(self, inputs, encoder_hidden_state, projected_keys, encoder_outputs, teacher_forcing=.4):
        """
        The following takes in the entirety of the inputs for the
        decoder, passes it to our stepwise function, lexical_step(),
        and iterates through the sentence to create a useful output.
        """
        hidden, outputs, alphas_out, pre_outputs = encoder_hidden_state, [], [], []

        # 10/26 Changed loss such that true_y starts at n+1 ... this
        # was done in order to allow the model to start at decoder
        # inputs[0].
        xi = torch.tensor(inputs[0].item())
        for i in range(len(inputs)-1):
            output, hidden, pre_output, alphas = self.lexical_step(xi, hidden, projected_keys, encoder_outputs)
            output = self.outlayer(output)

            outputs.append(output)
            pre_outputs.append(pre_output)
            alphas_out.append(alphas)

            #implements a probabilistic teacher forcing shedule
            if np.random.rand() >= teacher_forcing:
                xi = torch.tensor(self.cos(output.view(1,-1), self.embeds).topk(k=1, dim=-1)[1].item())

            else:
                xi = torch.tensor(inputs[i+1].item())

        return torch.cat(outputs, dim=1), torch.cat(pre_outputs, dim=1), alphas_out, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class dec(nn.Module):

    def __init__(self, device, attention, embeddings, outputlayer, hidden_size=300, context_size=600, dropout_p=.1, bidirectional=True, SOS_token=False):
        super(dec, self).__init__()

        self.cos = nn.CosineSimilarity(dim=-1)
        self.device = device
        self.n_layers = 2
        self.hidden_size = hidden_size
        self.n_classes = len(embeddings) + 1
        self.bi = bidirectional
        self.output_size = self.hidden_size

        self.SOS = SOS_token

        self.bi_mul = 1
        if self.bi:
            self.bi_mul = 2

        #Embedding layer
        self.embeds = torch.FloatTensor(embeddings)
        #embed_size = self.embedding.shape[-1]
        #self.embedding = nn.Embedding.from_pretrained(self.embeds)
        #self.embedding = nn.Embedding(len(embeddings), len(embeddings[0]))
        self.embed_size = len(embeddings[0])
        #self.embeds = embeddings

        #Layers
        self.attention = attention
        self.outlayer = outputlayer
        self.dropout = nn.Dropout(p=dropout_p)
        self.decoder = nn.GRU(self.embed_size, self.hidden_size,
                              bidirectional=self.bi,
                              batch_first=True,
                              num_layers=2)

        self.pre_output_layer = nn.Linear(context_size, self.hidden_size)
        self.relu = nn.LeakyReLU()

    def lexical_step(self, input, hidden, projected_keys, encoder_outputs):
        """
        The following is a step-wise function that takes some input,
        the projected versions of the encoder states (states are reduced
        to hidden_size-D using the projection layer prima facie), the
        hidden state from the prior step, and the actual encoder_outputs
        to create a representation of the word going into the decoder
        and its context at t=sentence.index(word)
        """
        hidden_states = []

        #(1) Convert input to an embedding.
        embedded = self.embeds[input.item()].view(1,1,-1)
        embedded = self.dropout(embedded)


        # (3) GRU-1 Ops
        # (3.1) Concatenate the embedding rep of the word with the
        #       generated context vector.
        #gru1_input = torch.cat([embedded, context], dim=-1)
        # (3.2) Pass the previous concatenation as input to the
        #       GRU-1 layer
        output, hidden_states = self.decoder(embedded, hidden)

        # (2) Generate the context vector and alphas using the attention
        #    layer provided to the network on initialization.
        context, alphas = self.attention(output, projected_keys, encoder_outputs)
        output = self.dropout(self.relu(context))

        # (3.3) Apply relu function and dropout to output, in order to
        #       avoid disappearing gradients and overfiting,
        #       respectively.
        output = self.dropout(self.relu(output))

        # (5) Output Manipulation
        # (5.1) Concatenate embedding, output, and context vectors
        #       to be passed to the softmax, output layer. This gives
        #       our output access to all information pertaining to
        #       this time-step.
        #pre_output = torch.cat([output, context], dim=-1)
        # (5.2) Optionally, apply dropout to (5.1).
        pre_output = self.dropout(output)
        # (5.3) Now, compress this information to hidden_size, and
        #       simultaneously allow for a grad and differentiation to
        #       be applied here!
        pre_output = self.relu(self.pre_output_layer(pre_output))

        return output, hidden_states, pre_output, alphas

    def forward(self, inputs, encoder_hidden_state, projected_keys, encoder_outputs, teacher_forcing=.4):
        """
        The following takes in the entirety of the inputs for the
        decoder, passes it to our stepwise function, lexical_step(),
        and iterates through the sentence to create a useful output.
        """
        hidden, outputs, alphas_out, pre_outputs = encoder_hidden_state, [], [], []

        # 10/26 Changed loss such that true_y starts at n+1 ... this
        # was done in order to allow the model to start at decoder
        # inputs[0].
        xi = torch.tensor(inputs[0].item())
        for i in range(len(inputs)-1):
            output, hidden, pre_output, alphas = self.lexical_step(xi, hidden, projected_keys, encoder_outputs)
            output = self.outlayer(output)

            outputs.append(output)
            pre_outputs.append(pre_output)
            alphas_out.append(alphas)

            #implements a probabilistic teacher forcing shedule
            if np.random.rand() >= teacher_forcing:
                xi = torch.tensor(self.cos(output.view(1,-1), self.embeds).topk(k=1, dim=-1)[1].item())

            else:
                xi = torch.tensor(inputs[i+1].item())

        return torch.cat(outputs, dim=1), torch.cat(pre_outputs, dim=1), alphas_out, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
