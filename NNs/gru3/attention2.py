import torch
import torch.nn as nn

class bahdanau(nn.Module):

    def __init__(self, key_size, query_size, hidden_size):
        super(bahdanau, self).__init__()
        self.hidden_size = hidden_size
        self.proj_layer = nn.Linear(key_size, hidden_size)
        self.query_layer = nn.Linear(query_size, hidden_size)
        #self.reduction_layer = nn.Linear(hidden_size*2, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.alphas = None

    def forward(self, hidden, keys, encoder_outputs):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        #(1) Construct query vector after flattenting hidden states
        # (1.1) Flatten hidden states
        #query_input = torch.cat(hidden, dim=-1).view(-1)

        # (1.2) pass to the query layer NN realization
        query = self.query_layer(hidden)


        #(3) Combine query & key, and then pass those to energy_layer.
        alphas = self.energy_layer(torch.tanh(query + keys))
        alphas = self.softmax(alphas)
        alphas = alphas.transpose(-1, 1)

        #(4) Multiply and sum everything.
        values = encoder_outputs.unsqueeze(0) #torch.transpose(encoder_outputs, -1, 0).unsqueeze(0)
        context = alphas @ values
        context = context.view(1, 1, -1) #torch.transpose(context.squeeze(0), -1, 0).view(1, 1, -1)

        return context, alphas


class rosen(nn.Module):

    def __init__(self, key_size, query_size, hidden_size):
        super(rosen, self).__init__()
        self.hidden_size = hidden_size
        self.proj_layer = nn.Linear(key_size, hidden_size)
        self.query_layer = nn.Linear(query_size, hidden_size)
        #self.reduction_layer = nn.Linear(hidden_size*2, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.alphas = None

    def forward(self, hidden, keys, encoder_outputs):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        #(1) Construct query vector after flattenting hidden states
        # (1.1) Flatten hidden states
        #query_input = torch.cat(hidden, dim=-1).view(-1)

        # (1.2) pass to the query layer NN realization
        query = self.query_layer(hidden)


        #(3) Combine query & key, and then pass those to energy_layer.
        alphas = self.energy_layer(torch.tanh(query + keys))
        alphas = self.softmax(alphas)
        alphas = alphas.view(-1,1)#alphas.transpose(-1, 1)

        #(4) Multiply and sum everything.
        values = encoder_outputs #torch.transpose(encoder_outputs, -1, 0).unsqueeze(0)
        #print(values.shape, alphas.shape)
        context = (alphas * values).mean(dim=0).unsqueeze(0) #This will implement a harder selection . . . likely zeroing in on the exactly correct option.
        context = context.view(1, 1, -1) #torch.transpose(context.squeeze(0), -1, 0).view(1, 1, -1)

        return context, alphas


class cosR(nn.Module):
    """
    Built on:
    03142020
    """

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=1, local_noise=.8, cosine_threshold=.4):
        super(cosR, self).__init__()
        self.hidden_size = hidden_size
        self.novelty_bias = novelty_bias

        self.cos = nn.CosineSimilarity(dim=-1)
        self.cosEl = nn.CosineSimilarity(dim=1)
        self.alpha_l1 = nn.Linear(2, 2, bias=False)
        self.alpha_l2 = nn.Linear(hidden_size, 1, bias=False)
        self.alpha_threshold = nn.Threshold(cosine_threshold, 0.)
        self.alphaMax = nn.Softmax(dim=0)

        # Layers
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.l1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.l2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.l3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

        # Non linear activations
        self.softmaxEl = nn.Softmax(dim=-1)
        self.softmaxSeq = nn.Softmax(dim=1)
        self.softmaxTout = nn.Softmax(dim=0)
        self.sig = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(.01)
        self.rrelu = nn.RReLU(.01, .1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.current_drop = nn.Dropout(local_noise)
        self.memory_drop = nn.Dropout(.3)

        self.alphas = None

    def forward(self, pre_outputs, memory):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        # (1) Create Alpha layer from Cosine Similarity of current pre_outputs
        #    and global memory.
        alphas = (self.alpha_threshold(self.cos(pre_outputs, memory)) > 0.).float()
        alphas = self.alphaMax(self.alpha_l2(alphas.unsqueeze(-1) * memory))

        # (2)
        shared_context = alphas * memory
        shared_context = self.softmaxSeq(self.l1(shared_context))
        shared_context = self.softmaxEl(self.l2(shared_context))
        shared_context = self.relu(self.l3(shared_context))
        shared_context = torch.tanh(shared_context.mean(dim=0).unsqueeze(0))

        # (3)
        discount = self.current_drop(pre_outputs * self.novelty_bias)
        context = self.relu(
            self.out(torch.tanh(shared_context + discount)))  # self.relu(self.out(torch.tanh(context + discount)))
        return context, context


class cosR3(nn.Module):
    """
    Built on:
    03142020
    """

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=1, local_noise=.5, cosine_threshold=.4):
        super(cosR3, self).__init__()
        self.hidden_size = hidden_size
        self.novelty_bias = novelty_bias

        self.cos = nn.CosineSimilarity(dim=-1)
        self.alpha_l2 = nn.Linear(hidden_size, 1, bias=False)
        self.alpha_l1 = nn.Linear(2, 2, bias=False)
        self.alpha_threshold = nn.Threshold(cosine_threshold, 0.)
        self.alphaMax = nn.Softmax(dim=0)

        self.discount_threshold = .15 #nn.Threshold(.15, 0.)

        # Layers
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

        # Non linear activations
        self.softmaxEl = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.current_drop = nn.Dropout(local_noise)
        self.alphas = None

    def forward(self, pre_outputs, memory):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        # (1) Create Alpha layer from Cosine Similarity of current pre_outputs
        #    and global memory.
        alphas = self.cos(pre_outputs, memory)  # self.alpha_threshold(self.cos(pre_outputs, memory))
        #alphas = self.alphaMax(self.alpha_l2(alphas.unsqueeze(-1) * memory))
        alphas = self.alphaMax(self.alpha_l1(alphas).unsqueeze(-1))

        # (2)
        shared_context = alphas * memory
        context = (shared_context.mean(dim=0).unsqueeze(0))

        # (3)
        discount = self.softmaxEl(self.current_drop(pre_outputs))
        context = self.relu(self.out((context * discount)))
        return context, self.softmaxEl(shared_context.sum(dim=0)).view(pre_outputs.shape)


class cosR4(nn.Module):
    """
    Built on:
    03142020
    """

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=1, local_noise=.5, cosine_threshold=.4):
        super(cosR4, self).__init__()
        self.hidden_size = hidden_size
        self.novelty_bias = novelty_bias

        self.cos = nn.CosineSimilarity(dim=-1)
        self.alpha_l1 = nn.Linear(2, 2, bias=False)
        self.alpha_l2 = nn.Linear(hidden_size, 1, bias=False)
        self.alpha_threshold = nn.Threshold(cosine_threshold, 0.)
        self.alphaMax = nn.Softmax(dim=0)

        self.discount_threshold = .15 #nn.Threshold(.15, 0.)

        # Layers
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

        # Non linear activations
        self.softmaxEl = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.current_drop = nn.Dropout(local_noise)
        self.alphas = None

    def forward(self, pre_outputs, memory):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        # (1) Create Alpha layer from Cosine Similarity of current pre_outputs
        #    and global memory.
        alphas = self.cos(pre_outputs, memory) #self.alpha_threshold(self.cos(pre_outputs, memory))
        alphas = self.alphaMax(self.alpha_l1(alphas).unsqueeze(-1))

        # (2)
        shared_context = alphas * memory
        context = (shared_context.mean(dim=0).unsqueeze(0))

        # (3)
        discount = self.softmaxEl(self.current_drop(pre_outputs)) * self.novelty_bias
        context = self.relu(self.out((context + discount)))
        return context, self.softmaxEl(shared_context.sum(dim=0)).view(pre_outputs.shape)


class cosR5(nn.Module):
    """
    Built on:
    03142020
    """

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=1, local_noise=.5, cosine_threshold=.4):
        super(cosR5, self).__init__()
        self.hidden_size = hidden_size
        self.novelty_bias = novelty_bias

        self.cos = nn.CosineSimilarity(dim=-1)
        self.alpha_l1 = nn.Linear(2, 2, bias=False)
        self.alpha_l2 = nn.Linear(hidden_size, 1, bias=False)
        self.alpha_threshold = nn.Threshold(cosine_threshold, 0.)
        self.alphaMax = nn.Softmax(dim=0)

        self.discount_threshold = .15 #nn.Threshold(.15, 0.)

        # Layers
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

        # Non linear activations
        self.softmaxEl = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.current_drop = nn.Dropout(local_noise)
        self.alphas = None

    def forward(self, pre_outputs, memory):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        # (1) Create Alpha layer from Cosine Similarity of current pre_outputs
        #    and global memory.
        noisy_input = self.current_drop(pre_outputs)
        alphas = self.cos(noisy_input, memory) #self.alpha_threshold(self.cos(pre_outputs, memory))
        alphas = self.alphaMax(self.alpha_l1(alphas).unsqueeze(-1))

        # (2)
        shared_context = alphas * memory
        context = (shared_context.mean(dim=0).unsqueeze(0))

        # (3)
        discount = self.softmaxEl(noisy_input)
        context = self.relu(self.out((context * discount)))
        return context, self.softmaxEl(shared_context.sum(dim=0)).view(pre_outputs.shape)

"""class R7(nn.Module):

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=1, seq_size=3, memory_threshold=.4, current_noise=.7):
        super(R7, self).__init__()
        self.hiddn_size=hidden_size
        self.hidden_size = hidden_size
        self.novelty_bias = novelty_bias
        self.seq_size = seq_size
        self.current_drop = nn.Dropout(current_noise)

        #Memory/Episode level operations
        self.cosimM = nn.CosineSimilarity(dim=-1)
        self.thM = nn.Threshold(memory_threshold, 0)
        self.actM = nn.Softmax(dim=1)
        self.alpha_M = nn.Linear(seq_size, seq_size, bias=False)

        #Element level operations
        self.cosimE = nn.CosineSimilarity(dim=1)
        self.actE = nn.Sigmoid()
        self.alpha_E = nn.Linear(hidden_size, hidden_size, bias=False)

        # Miscelaneous required
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, c, m):

        l1 = self.thM(self.cosimM(c,m))
        l1 = self.actM(self.alpha_M(l1))

        l2 = self.cosimE(c, m).unsqueeze(1).repeat_interleave(self.seq_size, 1)
        l2 = self.actE(self.alpha_E(l2))

        discount = self.current_drop(c * self.novelty_bias)
        shared_context = (l1.unsqueeze(-1) * m) + (l2 - discount)
        shared_context = shared_context.mean(0).view(c.shape)

        context = self.relu(self.out(shared_context))

        return context, context


class R8(nn.Module):

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=1, seq_size=3, memory_threshold=.4, element_threshold=.5):
        super(R8, self).__init__()
        self.hiddn_size = hidden_size
        self.hidden_size = hidden_size
        self.novelty_bias = novelty_bias
        self.seq_size = seq_size

        #Memory/Episodic level operations
        self.cosimM = nn.CosineSimilarity(dim=-1)
        self.thM = nn.Threshold(memory_threshold, 0.)
        self.actM = nn.Sigmoid() #nn.Softmax(dim=1)
        self.alpha_M = nn.Linear(seq_size, seq_size, bias=False)

        #Element level operations
        self.cosimE = nn.CosineSimilarity(dim=1)
        self.thE = nn.Threshold(element_threshold, 0.)
        self.actE = nn.Sigmoid()
        self.alpha_E = nn.Linear(hidden_size, hidden_size, bias=False)

        #Miscelaneous required
        self.l1 = nn.Linear(1,1, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, c, m):

        l1 = self.cosimM(c,m)
        l1 = self.thM(l1)

        l2 = self.cosimE(c, m * l1.unsqueeze(-1))
        l2 = torch.bmm(m, l2.unsqueeze(1).transpose(1,-1))
        l2 = self.l1(l2)
        l2 = self.actE(l2)

        #l2 = self.cosimE(c, l1.unsqueeze(-1) * m)
        #l2 = self.actE(self.alpha_E(l2))
        #print(l1.shape, l2.shape)

        shared_context = (m * l2)
        shared_context = shared_context.mean(0).view(c.shape) #shared_context.mean(0).view(c.shape)
        context = self.relu(self.out(shared_context))

        return context, context


class R9(nn.Module):

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=1, seq_size=3, memory_threshold=.4, element_threshold=.5):
        super(R9, self).__init__()
        self.hiddn_size = hidden_size
        self.hidden_size = hidden_size
        self.novelty_bias = novelty_bias
        self.seq_size = seq_size

        #Memory/Episodic level operations
        self.cosimM = nn.CosineSimilarity(dim=-1)
        self.thM = nn.Threshold(memory_threshold, 0.)
        self.actM = nn.Sigmoid() #nn.Softmax(dim=1)
        self.alpha_M = nn.Linear(seq_size, seq_size, bias=False)

        #Element level operations
        self.cosimE = nn.CosineSimilarity(dim=1)
        self.thE = nn.Threshold(element_threshold, 0.)
        self.actE = nn.Sigmoid()
        self.alpha_E = nn.Linear(hidden_size, hidden_size, bias=False)

        #Miscelaneous required
        self.l1 = nn.Linear(1,1, bias=False)
        self.gru = nn.GRU(hidden_size*3, hidden_size*3, batch_first=True, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, c, m):

        l1 = self.cosimM(c,m)
        l1 = self.thM(l1)

        l2 = self.cosimE(c, m * l1.unsqueeze(-1))
        l2 = torch.bmm(m, l2.unsqueeze(1).transpose(1,-1))
        l2 = self.l1(l2)
        l2 = self.actE(l2)

        l3 = (l2 * m).view(1, -1, m.shape[1]*m.shape[2])
        _, shared_context = self.gru(l3)

        #l2 = self.cosimE(c, l1.unsqueeze(-1) * m)
        #l2 = self.actE(self.alpha_E(l2))
        #print(l1.shape, l2.shape)

        #shared_context = (m * l2)
        #shared_context = shared_context.mean(0).view(c.shape) #shared_context.mean(0).view(c.shape)
        shared_context = shared_context.view(c.shape)
        context = self.relu(self.out(shared_context))

        return context, context


class r10(nn.Module):

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=.01, sequence_size=2, memory_threshold=.8, element_threshold=.7, forgetfulness=.5):
        super(r10,self).__init__()
        self.hdn_size = hidden_size
        self.seq_size = sequence_size
        self.c_input = nn.Linear(hidden_size, hidden_size, bias=False)
        self.m_input = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relu = nn.ReLU()

        self.noise = novelty_bias
        self.current_representation_noise = nn.Dropout(forgetfulness)

        self.cos = nn.CosineSimilarity(dim=-1)
        self.mem_threshold = memory_threshold#nn.Threshold(memory_threshold, 0.)
        self.el_threshold = element_threshold#nn.Threshold(element_threshold, 0.)

        self.mem_l1 = nn.Linear(2, 2, bias=False)
        self.mem_l2 = nn.Linear(20, 100, bias=False)
        self.mem_l3 = nn.Linear(100, 1, bias=False)
        self.mem_gru1 = nn.GRU(hidden_size*sequence_size, hidden_size*sequence_size, bias=False, batch_first=True)
        self.mem_gru2 = nn.GRU(hidden_size*2, hidden_size*sequence_size, bias=False, batch_first=True)
        self.mem_sigmoid = nn.Sigmoid()
        self.mem_softmax = nn.Softmax(dim=0)


        self.cosE = nn.CosineSimilarity(dim=1)
        self.E = nn.Linear(hidden_size, hidden_size, bias=False)
        self.SoftmaxE = nn.Softmax(dim=1)
        self.softmaxE = nn.Softmax(dim=0)

        self.l1 = nn.Linear(hidden_size, hidden_size*2, bias=False)
        self.l2 = nn.Linear(hidden_size*2, hidden_size*2, bias=False)
        self.out = nn.Linear(hidden_size*2, hidden_size, bias=False)

    def forward__(self, c, m):
        noisy_input = self.c_input(self.current_representation_noise(c*self.noise))
        call = self.softmaxE(self.cosE(noisy_input, m).unsqueeze(1) @ m.transpose(1, -1))
        call = call * m

        out = self.l1(torch.tanh(call.sum(dim=0).view(c.shape)))
        out = self.relu(out+noisy_input)
        out = self.relu(self.out(out))

        return out, out

    def forward(self, c, m):
        #Note: Trying right now to instantiate a simple signal
        #      signal boosting algorithm from the 2D cos_sim
        #      rankings. A better method might be to make this
        #      recurrent. It's small data wise, and should be
        #      quick to iterate over.
        #seq = self.relu(self.c_input(self.current_representation_noise(c)))
        #seq = (self.cos(c, m).unsqueeze(-1) * m).view(1, -1, self.hdn_size*c.shape[1])
        #deltaS = (self.cos(c,m) >= self.mem_threshold).float().unsqueeze(-1)
        deltaS = self.cos(c,m)
        deltaS = self.mem_sigmoid(self.mem_l1(deltaS)).unsqueeze(-1)

        deltaE = self.cosE(c,m) #(self.cosE(c,m) >= self.el_threshold).float()
        deltaE = self.mem_sigmoid(self.E(deltaE))
        deltaE = deltaE.unsqueeze(1).repeat_interleave(c.shape[1], 1)

        seq = (m * deltaS * deltaE).view(1, -1, c.shape[1]*c.shape[-1])
        #seq, h = self.mem_gru1(seq)
        _, seq = self.mem_gru1(seq)
        #_, seq = self.mem_gru2(seq)
        seq = self.relu(seq).view(c.shape)

        out = self.relu(self.l1(seq))
        out = self.relu(self.l2(out))
        out = self.relu(self.out(out))

        return out, out


class r11(nn.Module):

    def __init__(self, key_size, query_size, hidden_size, forgetfulness=.5, memory_similarity=.8, element_similarity=.8, novelty_bias=.01):
        super(r11, self).__init__()
        self.hdn_size = hidden_size
        self.relu = nn.ReLU()
        self.novelty_bias = novelty_bias

        self.c_representation_noise = nn.Dropout(forgetfulness)
        self.c_input = nn.Linear(hidden_size, hidden_size, bias=False)


        self.cos = nn.CosineSimilarity(dim=-1)
        self.mem_threshold = nn.Linear(2,2, bias=False) #nn.Threshold(memory_similarity, 0.)
        self.mAct = nn.Softmax(dim=0)

        self.cosE = nn.CosineSimilarity(dim=1)
        self.el_threshold = nn.Linear(hidden_size, hidden_size, bias=False) #nn.Threshold(element_similarity, 0.)
        self.eAct = nn.Softmax(dim=0)

        #Autoencoder
        self.encoder = nn.Sequential(nn.Linear(hidden_size, int(hidden_size/2)),
                                      nn.Linear(int(hidden_size/2), int(hidden_size/4)))

        self.decoder = nn.Sequential(nn.Linear(int(hidden_size / 4), int(hidden_size / 2)),
                                      nn.Linear(int(hidden_size / 2), hidden_size),
                                     nn.Sigmoid())

        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, c, m):
        #Note: Trying right now to instantiate a simple signal
        #      signal boosting algorithm from the 2D cos_sim
        #      rankings. A better method might be to make this
        #      recurrent. It's small data wise, and should be
        #      quick to iterate over.
        noisy_input = self.relu(self.c_input(self.c_representation_noise(c)))

        deltaS = self.mAct(self.mem_threshold(self.cos(c,m))).unsqueeze(-1)#(self.mem_threshold(self.cos(c,m)) > 0.).float().unsqueeze(-1)
        seq = m * deltaS

        deltaE = self.eAct(self.el_threshold(self.cosE(c,m))) #(self.el_threshold(self.cosE(c, m)) > 0.).float()
        el = deltaE.unsqueeze(1).repeat_interleave(c.shape[1], 1)

        seq = seq * el
        seq = (seq - noisy_input).mean(0).view(c.shape)

        out = self.relu(self.encoder(seq))
        out = self.decoder(out)
        out = self.relu(self.out(out))

        return out, out"""

