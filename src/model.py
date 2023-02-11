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

        