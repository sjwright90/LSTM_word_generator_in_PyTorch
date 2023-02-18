
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src import WordGenerator
from utils import Preprocessing
from utils import parameter_parser

class Execution:

    def __init__(self,args):
        self.file = args.file
        self.window = args.window
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        self.char_gen = args.char_gen
        self.model_save = args.model

        self.targets = None
        self.sequences = None
        self.vocab_size = None
        self.char_to_idx = None
        self.idx_to_char = None

        if torch.has_mps:
            self.device = torch.device("mps")
            print("gpu active")
        elif torch.has_cuda:
            self.device = torch.device("cuda")
            print("cuda active")
        else:
            self.device = torch.device("cpu")
            print("no gpu")

    def prepare_data(self):

        # initialize preprocessor object
        preprocessing = Preprocessing()

        # file loaded and split by char or word
        # depending on input
        if self.char_gen:
            text = preprocessing.read_file_char(self.file)
        else:
            text = preprocessing.read_file_word(self.file)

        # create two dictionaries from the text
        # char_to_idx and idx_to_char
        self.char_to_idx, self.idx_to_char = preprocessing.create_dictionary(text)
            

        # given a window, create training sequences
        # as well as targets
        self.sequences, self.targets = preprocessing.seq_target(text, self.char_to_idx, window=self.window)

        # get vocab size
        self.vocab_size = len(self.char_to_idx)
    
    def train(self, args):

        # Initialize model
        model = WordGenerator(args, self.vocab_size)
        model.to(device=self.device)

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4, amsgrad=True)

        # Defining number of batches
        num_batches = int(len(self.sequences)/self.batch_size)

        # set model in training mode
        model.train()

        loss_log = [np.inf]

        for epoch in range(self.num_epochs):
            # mini batches
            for i in range(num_batches):

                # batch definition
                try:
                    x_batch = self.sequences[i * self.batch_size : (i+1) * self.batch_size]
                    y_batch = self.targets[i * self.batch_size : (i+1) * self.batch_size]
                except:
                    x_batch = self.sequences[i * self.batch_size :]
                    y_batch = self.targets[i * self.batch_size :]
                    
                # convert numpy array to Torch tensor

                x = torch.from_numpy(x_batch).type(torch.LongTensor).to(device=self.device)
                y = torch.from_numpy(y_batch).type(torch.LongTensor).to(device=self.device)

                # feed the model
                y_pred = model(x)

                # calculate loss
                loss = F.cross_entropy(y_pred, y.squeeze())
                # clean the gradients
                optimizer.zero_grad()
                # back propogate
                loss.backward()
                # clip gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                # update parameters
                optimizer.step()
            print("Epoch: %d,  loss: %.5f " % (epoch, loss.item()))
            loss_score = loss.item()

            # if model decays break
            if np.isnan(loss_score):
                break

            # append loss each epoch
            loss_log = loss_log + [loss_score]

            # save state dictionary if loss decreases
            if loss_log[-1] < loss_log[-2]:
                torch.save(model.state_dict(), self.model_save)

            # early stopping if average loss over 3 epochs does not change
            if len(loss_log) > 5:
                if round(sum(loss_log[-1:-4:-1])/3,2) == round(loss_log[-1],2):
                    break
        
    @staticmethod
    def generator(model, sequences, idx_to_char, n_chars, char_gen):

        if torch.has_mps:
            device = torch.device("mps")
        elif torch.has_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        model.to(device=device)

        # set model in eval mode
        model.eval()

        # define softmax fnc
        softmax = nn.Softmax(dim=1)

        # ramdomly select index from set of sequences
        start = np.random.randint(0, len(sequences) - 1)

        # pattern is defined from start
        pattern = sequences[start]

        #use dictionaries to print the pattern
        if char_gen:
            print("\nPattern: \n")
            print("".join(idx_to_char[value] for value in pattern), "\"")
        else:
            print("\nPattern: \n")
            print("".join(idx_to_char[value] + " " for value in pattern), "\"")

        # In full_prediction we will save the complete prediction
        full_prediction = pattern.copy()

        # The prediction starts, it is going to be predicted a given
        # # number of characters

        for i in range(n_chars):

            # The numpy patterns is transformed into a tesor-type and reshaped
            pattern = torch.from_numpy(pattern).type(torch.LongTensor).to(device=device)
            pattern = pattern.view(1,-1)

            # Make prediction given pattern
            prediction = model(pattern)
            # apply softmax to prediction tensor
            prediction = softmax(prediction)
            

            # convert prediction tensor to numpy array
            prediction = prediction.to("cpu")
            prediction = prediction.squeeze().detach().numpy()
            
            # randomly select best choice from probability distribution
            # of predicted classes
            # if probability distribution is not appropriate
            # for np.random.choice then take highest probability
            try:
                arg_max = np.random.choice(list(idx_to_char.keys()), p = prediction)
            except:
                arg_max = np.argmax(prediction)

            # The current pattern tensor is transformed into numpy array
            pattern = pattern.to("cpu")
            pattern = pattern.squeeze().detach().numpy()

            # The window is sliced 1 character to the right
            pattern = pattern[1:]

            # The new pattern is composed by the "old" pattern + the predicted character
            pattern = np.append(pattern, arg_max)

            # The full prediction is saved
            full_prediction = np.append(full_prediction, arg_max)

        if char_gen:
                print("Prediction: \n")
                print("".join([idx_to_char[value] for value in full_prediction]), "\"")
        else:
            print("Prediction: \n")
            print("".join([idx_to_char[value] + " " for value in full_prediction]), "\"")


if __name__ == "__main__":

        args = parameter_parser()

        # If you alreeady have trained weights
        if args.load_model == True:
            if os.path.exists(args.model):

                # load and prepare sequence
                execution = Execution(args)
                execution.prepare_data()

                sequences = execution.sequences
                idx_to_char = execution.idx_to_char
                vocab_size = execution.vocab_size

                # initialize the model
                model = WordGenerator(args, vocab_size)
                # load weights
                model.load_state_dict(torch.load(args.model))

                # text generator
                execution.generator(model, sequences, idx_to_char, args.num_predict, args.char_gen)
        
        # if you will train the model
        else:
            # load and prep sequence
            execution = Execution(args)
            execution.prepare_data()

            # Training model
            execution.train(args)

            sequences = execution.sequences
            idx_to_char = execution.idx_to_char
            vocab_size = execution.vocab_size

            # initialize model
            model = WordGenerator(args, vocab_size)
            #load weights
            model.load_state_dict(torch.load(args.model))

            # text generator
            execution.generator(model, sequences, idx_to_char, args.num_predict, args.char_gen)