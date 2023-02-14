# Chasing the Whale: A Bi-Directional LSTM to write Melville

This repository is an implementation of an LSTM recurrent neural network built using PyTorch. The model architecture consists of a bi-directional layer, followed by three LSTM layers with dropout layers between the second and third LSTM layer, finally the results are passed to a linear layer. 

The scripts are set up to automatically move tensors onto and off of a GPU and will check the machine for the appropriate GPU backend (CUDA or MPS). 

The model uses cross entropy loss, an Adam optimizer, and gradient clipping is applied between the loss.backward() call and the optimizer.step() call. 

Model prediction is made using a random selection from the probability distribution of softmax normalized model outputs.

Additionally, the model is set up to allow the user to specify if they want to train a character or word model. Character models will include all alphabetical characters and the "." character, word models will include all words in the text and the "." character. 

The model is run through the command line with the following prompt:
```
 % Python3 -B main.py [--epochs NUM_EPOCHS] [--learning_rate LEARNING_RATE] [--hidden_dim HIDDEN_DIM] [--batch_size BATCH_SIZE] [--window WINDOW]
               [--load_model LOAD_MODEL] [--model MODEL] [--training_file FILE] [--char_gen CHAR_GEN] [--num_predict NUM_PREDICT]
```
A full description of input parameters can be found at the bottom of this README

The model is highly malleable just from the command line input, although certain parameters are hard coded, such as the model architecture and Adam weight_decay. One can changed these "fixed" values by going into the code base itself.

The training function is set up with two early stopping protocols. Early stopping will happen if any loss value comes back as NaN, a loss of NaN indicates that the model has diverged. The other early stopping kicks in when there is minimal change in loss over time. The loss per epoch is stored and starting at the 5th epoch if the average loss over the last three epochs equals the current loss (rounded to 2 significant figures) the model will stop training.

The model was trained on the first couple chapters of Herman Melville's Mboy-Dick. The text was downloaded from the Gutenburg Project (https://www.gutenberg.org/ebooks/2701) it can be downloaded directly from the command line using:
```
% curl https://www.gutenberg.org/files/2701/2701-0.txt --output FILE_NAME.txt
```

Despite extensive testing of architecture and parameters, I could not get great convergence from the model. Further tuning might improve the model, but as my purpose was to experiment with PyTorch LSTM architectures I feel I met that goal.

Example of model output:

![](https://github.com/sjwright90/LSTM_word_generator/blob/main/figs/output.png)

Note: much of the base code was borrowed from here: https://github.com/FernandoLpz/Text-Generation-BiLSTM-PyTorch to give credit where credit is due. I recommend checking out his repo and associated blog post as they are very insightful.


## Description of command line input:

main.py is the driver code
--epochs is the number of epochs to run the model for (accepts an int)
--learning_rate is the learning rate for the Adam optimizer (accepts a float)
--hidden_dim is the number of features for the hidden state of the LSTM (also maps to the size of the current state) (accepts an int)
--batch_size is the size of the batches (the number of batches will be calculated from the number of sequences divided by batch_size) (accepts an int)
--window is the sequence length (the number of trailing characters or letters used to calculate the next character or letter, the division of the document by this value will give the number of sequences) (accepts an int)
--load_model is a binary option to predict or train when load_model==True a model will be loaded and a prediction run when False a model will be trained then a prediction made (can use 0 for False and 1 for True)
--model is the path directory to the .pt file where the script will save and load the model weights accepts a file pathway as a string
--training_file is the text file to train or predict the model on (accepts a file pathway as a sting)
--char_gen tells the model to use characters or words the default is false which is a word model
--num_predict is the number of outputs to predict it (accepts an integer).
