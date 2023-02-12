import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WordGenerator(nn.Module):
    def __init__(self, args, vocab_size):
        super(WordGenerator, self).__init__()

        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.input_size = vocab_size
        self.num_classes = vocab_size
        self.sequence_len = args.window

        # make embeddings
        self.embeddings = nn.Embedding(vocab_size, self.hidden_dim, padding_idx=0)

        # make forwards and backwards layers

        self.lstm_forward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_backward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(p = 0.2)

        # make LSTM layers

        self.lstm_1 = nn.LSTMCell(self.hidden_dim*2, self.hidden_dim*2)
        self.lstm_2 = nn.LSTMCell(self.hidden_dim*2, self.hidden_dim*2)
        #self.lstm_3 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        #self.lstm_4 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        #self.lstm_5 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        # make linear layer
        self.linear = nn.Linear(self.hidden_dim*2, self.num_classes)



    def forward(self, x):
        if torch.has_mps:
            mpsdevice = torch.device("mps")

        out = self.embeddings(x)

        out = out.view(self.sequence_len, x.size(0), -1)

        # initialize states as zeros
        hs_backward = torch.zeros(x.size(0), self.hidden_dim).to(device=mpsdevice)
        cs_backward = torch.zeros(x.size(0), self.hidden_dim).to(device=mpsdevice)
        hs_forward = torch.zeros(x.size(0), self.hidden_dim).to(device=mpsdevice)
        cs_forward = torch.zeros(x.size(0), self.hidden_dim).to(device=mpsdevice)
        hs_lstm_1 = torch.zeros(x.size(0), self.hidden_dim*2).to(device=mpsdevice)
        cs_lstm_1 = torch.zeros(x.size(0), self.hidden_dim*2).to(device=mpsdevice)
        hs_lstm_2 = torch.zeros(x.size(0), self.hidden_dim*2).to(device=mpsdevice)
        cs_lstm_2 = torch.zeros(x.size(0), self.hidden_dim*2).to(device=mpsdevice)

        # initialize weights

        for lay in [hs_backward, cs_backward, hs_forward, cs_forward, hs_lstm_1, cs_lstm_1, hs_lstm_2, cs_lstm_2]:
            nn.init.xavier_uniform_(lay)
        
        forward = []
        backward =[]

        for i in range(self.sequence_len):
            hs_forward, cs_forward = self.lstm_forward(out[i], (hs_forward, cs_forward))
            forward.append(hs_forward)
        
        for i in reversed(range(self.sequence_len)):
            hs_backward, cs_backward = self.lstm_backward(out[i], (hs_backward, cs_backward))
            backward.append(hs_backward)
        
        for fwd, bwd in zip(forward, backward):
            in_tensor = torch.cat((fwd, bwd), 1)
            hs_lstm_1, cs_lstm_1 = self.lstm_1(in_tensor, (hs_lstm_1, cs_lstm_2))
            hs_lstm_1 = self.dropout(hs_lstm_1)
            hs_lstm_2, cs_lstm_2 = self.lstm_2(hs_lstm_1, (hs_lstm_2, cs_lstm_2))


        out = self.linear(hs_lstm_2)
        
        return out