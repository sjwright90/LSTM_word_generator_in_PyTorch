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

        # make LSTM layers

        self.lstm_1 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_3 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_4 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_5 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        # make linear layer
        self.linear = nn.Linear(self.hidden_dim, self.num_classes)

        if torch.has_mps:
            self.device = torch.device("mps")


    def forward(self, x):

        out = self.embeddings(x)

        out = out.view(self.sequence_len, x.size(0), -1)

        # initialize states as zeros
        hs_lstm_1 = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        cs_lstm_1 = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        hs_lstm_2 = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        cs_lstm_2 = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        hs_lstm_3 = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        cs_lstm_3 = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        hs_lstm_4 = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        cs_lstm_4 = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        hs_lstm_5 = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        cs_lstm_5 = torch.zeros(x.size(0), self.hidden_dim).to(self.device)

        # initialize weights
        for lay in [hs_lstm_1, cs_lstm_1, hs_lstm_2, cs_lstm_2, hs_lstm_3, cs_lstm_3, hs_lstm_4, cs_lstm_4, hs_lstm_5, cs_lstm_5]:
            nn.init.kaiming_normal_(lay)

        for i in range(self.sequence_len):
            hs_lstm_1, cs_lstm_1 = self.lstm_1(out[i], (hs_lstm_1, cs_lstm_1))
            hs_lstm_2, cs_lstm_2 = self.lstm_2(hs_lstm_1, (hs_lstm_2, cs_lstm_2))
            hs_lstm_3, cs_lstm_3 = self.lstm_3(hs_lstm_2, (hs_lstm_3, cs_lstm_3))
            hs_lstm_4, cs_lstm_4 = self.lstm_4(hs_lstm_3, (hs_lstm_4, cs_lstm_4))
            hs_lstm_5, cs_lstm_5 = self.lstm_5(hs_lstm_4, (hs_lstm_5, cs_lstm_5))

        out = self.linear(hs_lstm_5)
        
        return out