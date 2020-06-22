import torch
import torch.nn as nn
import torch.nn.functional as F

class bahdanau(nn.Module):

    def __init__(self, key_size, query_size, hidden_size):
        super(bahdanau, self).__init__()
        self.hidden_size = hidden_size
        self.proj_layer = nn.Linear(key_size, hidden_size)
        self.query_layer = nn.Linear(query_size, hidden_size)
        #self.reduction_layer = nn.Linear(hidden_size*2, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=0)
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
        query_input = torch.cat(hidden, dim=-1).view(-1)

        # (1.2) pass to the query layer NN realization
        query = self.query_layer(query_input)

        # (1.3) Since we have variable queryN v. encoder_ouputN
        #       (we're masking NOTHING!), we need to blow up the
        #       the number of query repetitions by the number of
        #       encoder outputs for the energy_layer.
        #query = query.repeat(encoder_outputs.size()[0]).view(-1, self.hidden_size)

        #(2) Take the encoder_outputs and create a key_layer
        ##### THIS MAY NEED TO BE MOVED OUTSIDE OF THE LOOP
        #keys = self.key_layer(encoder_outputs)

        #(3) Combine query & key, and then pass those to energy_layer.
        alphas = self.softmax(self.energy_layer(torch.tanh(query + keys))).transpose(0, -1).unsqueeze(0)

        #(4) Multiply and sum everything.
        values = encoder_outputs.unsqueeze(0) #torch.transpose(encoder_outputs, -1, 0).unsqueeze(0)
        context = torch.bmm(alphas, values)
        context = context.view(1, 1, -1) #torch.transpose(context.squeeze(0), -1, 0).view(1, 1, -1)

        return context, alphas

class cosBahdanau(nn.Module):

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=1):
        super(cosBahdanau, self).__init__()
        self.hidden_size = hidden_size
        self.novelty_bias = novelty_bias

        self.proj_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size)
        #V1
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        #EXPERIMENTAL self.energy_layer = nn.Linear(1, 1, bias=False)
        self.combinatory_layer = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=0)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.alphas = None

    def forward_local(self, hidden, keys, encoder_outputs):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        #(1) Construct query vector after flattenting hidden states
        # (1.1) Flatten hidden states
        query_input = torch.cat(hidden, dim=-1).view(-1)

        # (1.2) pass to the query layer NN realization
        query = self.query_layer(query_input)

        # (1.3) Since we have variable queryN v. encoder_ouputN
        #       (we're masking NOTHING!), we need to blow up the
        #       the number of query repetitions by the number of
        #       encoder outputs for the energy_layer.
        #query = query.repeat(encoder_outputs.size()[0]).view(-1, self.hidden_size)

        #(2) Take the encoder_outputs and create a key_layer
        ##### THIS MAY NEED TO BE MOVED OUTSIDE OF THE LOOP
        #keys = self.key_layer(encoder_outputs)

        #(3) Combine query & key, and then pass those to energy_layer.
        alphas = self.softmax(self.cos_sim(query.unsqueeze(0), keys)) #self.softmax(self.energy_layer(torch.tanh(query + keys))).transpose(0, -1).unsqueeze(0)

        #(4) Multiply and sum everything.
        values = encoder_outputs.unsqueeze(0) #torch.transpose(encoder_outputs, -1, 0).unsqueeze(0)
        context = torch.bmm(alphas.unsqueeze(0).unsqueeze(0), values)

        context = context.view(1, 1, -1) #torch.transpose(context.squeeze(0), -1, 0).view(1, 1, -1)

        return context, alphas

    def forward(self, pre_outputs, memory):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        #(1) Create Alpha layer from Cosine Similarity of current pre_outputs
        #    and global memory.
        alphas = self.cos_sim(pre_outputs, memory)
        #V1
        alphas = self.softmax(self.energy_layer(alphas.unsqueeze(-1)*memory))
        #EXPERIMENTAL alphas = self.softmax(self.energy_layer(alphas.unsqueeze(-1)))

        #(2) Dot Product of Alphas and Memory
        context = alphas * memory
        context = context.sum(dim=0)

        #(3) Enforce context layer's shape to match that of pre_outputs
        context = context.view(pre_outputs.shape)

        #(4) Add and pass summed representations to combinatory layer to
        #    render differentiable.
        #EXPERIMENTAL context = self.relu(self.combinatory_layer(torch.tanh(context)))
        #V1
        context = self.relu(self.combinatory_layer(torch.tanh(context+(pre_outputs*self.novelty_bias))))

        return context, alphas

class cosRosen03(nn.Module):
    """
    A near perfect implementation.
    03142020
    """

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=1, local_noise=.7):
        super(cosRosen03, self).__init__()
        self.hidden_size = hidden_size
        self.novelty_bias = novelty_bias
        self.cos_pct = nn.Linear(1, 1, bias=False)
        self.proj_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size)
        #ORIGINAL
        self.intermediary_layer = nn.Linear(hidden_size, hidden_size, bias=False)

        #EXPERIMENTAL
        self.energy_layer, self.combinatory_layer = nn.Linear(hidden_size, hidden_size, bias=False), nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmaxEl = nn.Softmax(dim=-1)
        self.softmaxSeq = nn.Softmax(dim=1)
        self.softmaxTout = nn.Softmax(dim=0)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.drop = nn.Dropout(local_noise)
        self.ctxt_drop = nn.Dropout(.3)

        self.alphas = None

    def forward(self, pre_outputs, memory):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        #(1) Create Alpha layer from Cosine Similarity of current pre_outputs
        #    and global memory.
        alphas = self.cos_sim(pre_outputs, memory)

        #(2) Dot Product of Alphas and Memory
        #alphas = self.softmax(self.cos_pct(alphas))

        #(3)
        shared_context = alphas.unsqueeze(dim=-1) * memory
        #shared_context = torch.tanh(shared_context.sum(dim=0))

        #(4) Create a softmax distribution over examples/sequences . . .
        context = self.softmaxTout(self.intermediary_layer(shared_context)) #context.mean(dim=0) #torch.tanh(context.sum(dim=0))

        #(5) Create a softmax distribution over elements in each item
        context = context * memory #shared_context
        context = self.softmaxEl(self.energy_layer(context))

        #(5.2 EXPERIMENTAL) Do not compress results until this last step. Operations roll
        #    over all items in memory, freely.
        context = context.sum(dim=0).view(pre_outputs.shape)
        context = torch.tanh(context)
        #context = self.ctxt_drop(context)

        #(6) Add and pass summed representations to combinatory layer to
        #    render differentiable.
        #context = self.combinatory_layer(torch.tanh(torch.tanh(context1 * context2 * shared_context))).view(pre_outputs.shape)
        #(!BEST)
        context = self.combinatory_layer(self.ctxt_drop(torch.tanh(context.view(pre_outputs.shape) + self.drop(pre_outputs * self.novelty_bias)))).view(pre_outputs.shape)
        #(NO CONTRIBUTION FROM CURRENT CONTEXT) context = self.combinatory_layer(context.view(pre_outputs.shape)).view(pre_outputs.shape)
        context = self.relu(context)

        return context, alphas

class cosRosen(nn.Module):

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=1):
        super(cosRosen, self).__init__()
        self.hidden_size = hidden_size
        self.novelty_bias = novelty_bias
        self.proj_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size)
        #ORIGINAL
        self.energy_layer = nn.Linear(hidden_size*2, hidden_size*2, bias=False)
        self.combinatory_layer = nn.Linear(hidden_size*2, hidden_size)

        self.softmax = nn.Softmax(dim=1)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.alphas = None

    def forward(self, pre_outputs, memory):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        #(1) Create Alpha layer from Cosine Similarity of current pre_outputs
        #    and global memory.
        alphas = self.cos_sim(pre_outputs, memory)

        #(2) Dot Product of Alphas and Memory
        context = alphas.unsqueeze(dim=-1) * memory
        context = context.mean(dim=0) #torch.tanh(context.sum(dim=0))

        #(3) Enforce context layer's shape to match that of pre_outputs
        context = torch.cat([(pre_outputs * self.novelty_bias), context.view(pre_outputs.shape)], dim=-1)

        #(4) Per Element calculation of importance to output.
        context = self.softmax(self.energy_layer(context))

        #(5) Add and pass summed representations to combinatory layer to
        #    render differentiable.
        context = self.relu(self.combinatory_layer(context))

        return context, alphas

class cosRosen06(nn.Module):
    """
    A near perfect implementation.
    03142020
    """

    def __init__(self, key_size, query_size, hidden_size, novelty_bias=1, local_noise=.8, cosine_threshold=.4):
        super(cosRosen06, self).__init__()
        self.hidden_size = hidden_size
        self.novelty_bias = novelty_bias

        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.cosEl = nn.CosineSimilarity(dim=1)
        self.alpha_l1 = nn.Linear(2, 2, bias=False)
        self.alpha_l2 = nn.Linear(hidden_size, 1, bias=False)
        self.alpha_threshold = nn.Threshold(cosine_threshold, 0.)
        self.alphaMax = nn.Softmax(dim=0)


        #Layers
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.l1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.l2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.l3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

        #Non linear activations
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

        #(1) Create Alpha layer from Cosine Similarity of current pre_outputs
        #    and global memory.
        alphas = self.alpha_threshold(self.cos_sim(pre_outputs, memory))
        #alphas = self.alphaMax(self.alpha_l1(alphas))
        alphas = self.alphaMax(self.alpha_l2(alphas.unsqueeze(-1) * memory))

        #(3)
        shared_context = alphas * memory
        shared_context = self.softmaxSeq(self.l1(shared_context))
        shared_context = self.softmaxEl(self.l2(shared_context))
        shared_context = self.relu(self.l3(shared_context))
        #shared_context = torch.tanh(shared_context.sum(dim=0).unsqueeze(0))

        discount = self.current_drop(pre_outputs * self.novelty_bias)
        context = torch.tanh(shared_context.mean(dim=0).unsqueeze(0))

        #multiplier = torch.bmm(self.current_drop(pre_outputs), context.transpose(1, -1)).sum(dim=1).unsqueeze(-1)
        #(w/  discount)
        #context = torch.cat([context, discount], dim=-1)
        #context = self.relu(self.out(context))
        context = self.relu(self.out(torch.tanh(context + discount))) #self.relu(self.out(torch.tanh(context + discount)))
        return context, context

class dot(nn.Module):

    def __init__(self, key_size, query_size, hidden_size):
        super(dot, self).__init__()
        #self.hidden_size = hidden_size
        #self.key_size = key_size
        #self.query_size = query_size
        self.l1 = nn.Linear(1, 1)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        #self.l3 = nn.Linear(hidden_size, hidden_size)
        #self.alpha_layer = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        #self.tanh = nn.Tanh()

        #PROJECT THIS LAYER OUTSIDE OF THE FORWARD LOOP . . . .
        # AS IN, CALL IT OVER ENCODER OUTPUTS PRIOR TO FORWARD
        #    proj_keys = attention.proj_layer(encoder_outputs)
        #    context, alphas = attention(hidden_, proj_keys)
        # THiS IS PROBABLY TO SAVE THE NUMBER OF TIMES THIS
        # GETS WEIGHTED/RECALCULATED.
        self.proj_layer = nn.Linear(key_size, hidden_size)

        #self.alphas = None

    def forward(self, hidden, projections, encoder_outputs):
        """ This function lives and dies off of having previously
         calculated the projection layer . . . this is run in
         training outside of the forward() call, post creation of
         the encoder outputs. """

        #(1) Construct query vector after flattenting hidden states
        # (1.1) Flatten hidden states
        query_input = sum(hidden)
        alphas = self.softmax(self.l1(torch.bmm(projections.unsqueeze(0), query_input.transpose(1, -1)))).transpose(1, -1)

        context = self.softmax(self.l2(torch.bmm(alphas, encoder_outputs.unsqueeze(0))))
        #context = self.relu(self.l1(context.squeeze(0)))
        #context = self.relu(self.l2(context))
        #context = self.relu(self.l3(context))

        """
        # (1.2) pass to the query layer NN realization
        query = self.query_layer(query_input)

        # (1.3) Since we have variable queryN v. encoder_ouputN
        #       (we're masking NOTHING!), we need to blow up the
        #       the number of query repetitions by the number of
        #       encoder outputs for the energy_layer.
        #query = query.repeat(encoder_outputs.size()[0]).view(-1, self.hidden_size)

        #(2) Take the encoder_outputs and create a key_layer
        keys = self.key_layer(encoder_outputs)

        #(3) Combine query & key, and then pass those to energy_layer.
        alphas = self.softmax(self.energy_layer(torch.tanh(query + keys))).transpose(0, -1).unsqueeze(0)

        #(4) Multiply and sum everything.
        values = encoder_outputs.unsqueeze(0) #torch.transpose(encoder_outputs, -1, 0).unsqueeze(0)
        context = torch.bmm(alphas, values)
        context = context.view(1, 1, -1) #torch.transpose(context.squeeze(0), -1, 0).view(1, 1, -1)
        """
        return context, alphas
