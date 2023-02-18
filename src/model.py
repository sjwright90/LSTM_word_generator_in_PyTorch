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

        if torch.has_mps:
            self.device = torch.device("mps")
            print("gpu active")
        elif torch.has_cuda:
            self.device = torch.device("cuda")
            print("cuda active")
        else:
            self.device = torch.device("cpu")
            print("no gpu")

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
        self.lstm_3 = nn.LSTMCell(self.hidden_dim*2, self.hidden_dim*2)


        # make linear layer
        self.linear = nn.Linear(self.hidden_dim*2, self.num_classes)

        # layer normalization
        self.lnorm = nn.LayerNorm(self.hidden_dim*2)



    def forward(self, x):

        out = self.embeddings(x)

        out = out.view(self.sequence_len, x.size(0), -1)

        # initialize states as zeros
        # move all tensors to correct device
        hs_backward = torch.zeros(x.size(0), self.hidden_dim).to(device=self.device)
        cs_backward = torch.zeros(x.size(0), self.hidden_dim).to(device=self.device)
        hs_forward = torch.zeros(x.size(0), self.hidden_dim).to(device=self.device)
        cs_forward = torch.zeros(x.size(0), self.hidden_dim).to(device=self.device)
        hs_lstm_1 = torch.zeros(x.size(0), self.hidden_dim*2).to(device=self.device)
        cs_lstm_1 = torch.zeros(x.size(0), self.hidden_dim*2).to(device=self.device)
        hs_lstm_2 = torch.zeros(x.size(0), self.hidden_dim*2).to(device=self.device)
        cs_lstm_2 = torch.zeros(x.size(0), self.hidden_dim*2).to(device=self.device)
        hs_lstm_3 = torch.zeros(x.size(0), self.hidden_dim*2).to(device=self.device)
        cs_lstm_3 = torch.zeros(x.size(0), self.hidden_dim*2).to(device=self.device)

        # initialize weights

        for lay in [hs_backward, cs_backward, hs_forward, cs_forward,
                    hs_lstm_1, cs_lstm_1,
                    hs_lstm_2, cs_lstm_2,
                    hs_lstm_3, cs_lstm_3]:
            nn.init.xavier_uniform_(lay)
        
        forward = []
        backward =[]

        for i in range(self.sequence_len):
            hs_forward, cs_forward = self.lstm_forward(out[i], (hs_forward, cs_forward))
            forward.append(hs_forward)
        
        for i in reversed(range(self.sequence_len)):
            hs_backward, cs_backward = self.lstm_backward(out[i], (hs_backward, cs_backward))
            backward.append(hs_backward)
        
        # dropout 0.2 between LSTM 2 and 3 layers
        # layer normalization between each layer
        for fwd, bwd in zip(forward, backward):
            in_tensor = torch.cat((fwd, bwd), 1)
            hs_lstm_1, cs_lstm_1 = self.lstm_1(in_tensor, (hs_lstm_1, cs_lstm_1))
            hs_lstm_2, cs_lstm_2 = self.lstm_2(hs_lstm_1, (hs_lstm_2, cs_lstm_2))
            hs_lstm_2 = self.dropout(hs_lstm_2)
            hs_lstm_3, cs_lstm_3 = self.lstm_3(hs_lstm_2, (hs_lstm_3, cs_lstm_3))

        # final dropout
        hs_lstm_3 = self.dropout(hs_lstm_3)
        # pass to linear layer
        out = self.linear(hs_lstm_3)
        
        return out