'''A python script for testing various functions from the preprocessing module'''
# %%
from utils import Preprocessing
# %%
out = Preprocessing.read_file_word("data/mobyshort.txt")
word, idx = Preprocessing.create_dictionary(out)
# %%
