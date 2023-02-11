# %%
import numpy as np
import re


class Preprocessing:

    @staticmethod
    def read_file(file):

        # read file in
        with open(file,"r") as f:
            raw_text = f.readlines()
    
        # make lower case
        raw_text = [line.lower() for line in raw_text]
        raw_text = [re.findall(r"\w+", line) for line in raw_text if len(line)>1]

        text = list()

        # concatenate to one long text file
        for line in raw_text:
            for word in line:
                if len(word) > 1:
                    text.append(word)

        # extract only words
    
        return text
    
    @staticmethod
    def create_dictionary(text):

        word_to_idx = dict(zip(set(text),
                           range(len(set(text)))))
        
        idx_to_word = dict(zip(word_to_idx.values(),
                               word_to_idx.keys()))

        return word_to_idx, idx_to_word

    @staticmethod
    def seq_target(text, char_to_idx, window):
        x = list()
        y = list()
        for i in range(len(text)):
            try:
                sequence = text[i:i+window]
                sequence = [char_to_idx[word] for word in sequence]

                target = text[i + window]
                target = char_to_idx[target]

                x.append(sequence)
                y.append(target)
            except:
                pass
        x = np.array(x)
        y = np.array(y)

        return(x,y)

