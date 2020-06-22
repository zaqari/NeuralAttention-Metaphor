import torch
import torch.nn as nn
import torch.nn.functional as F

class softmax_out(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(softmax_out, self).__init__()
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        output = self.out(x)
        output = self.softmax(output)
        return output

class logSoftmax_out(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(logSoftmax_out, self).__init__()
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        output = self.out(x)
        output = self.softmax(output)
        return output


class tanH_out(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(tanH_out, self).__init__()
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.out(x)
        output = torch.tanh(output)
        return output
