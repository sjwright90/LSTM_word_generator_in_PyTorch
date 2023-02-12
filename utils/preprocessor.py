# %%
import numpy as np
import re
import string


class Preprocessing:

    @staticmethod
    def read_file_word(file):

        # read file in
        with open(file,"r") as f:
            raw_text = f.readlines()
    
        # make lower case
        raw_text = [line.lower() for line in raw_text]
        raw_text = [re.findall(r"\w+|\.", line) for line in raw_text if len(line)>1]

        text = list()

        # concatenate to one long text file
        for line in raw_text:
            for word in line:
                if len(word) > 0:
                    text.append(word)

        # extract only words
        return text

    @staticmethod
    def read_file_char(file):
        letters = list(string.ascii_lowercase + " ")
        
        with open(file,"r") as f:
            raw_txt = f.readlines()
            
        raw_txt = [line.lower() for line in raw_txt]
        
        text_string = ""
        
        for line in raw_txt:
            text_string += line.strip()
    
        text = list()
        
        for char in text_string:
            text.append(char)
        
        text = [char for char in text if char in letters]
        
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


# %%
