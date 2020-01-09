## Implementation of a Hierarchical Attention Network (HAN)
##
## This implementation is based on the following paper
##
##   Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016, June). Hierarchical 
##     attention networks for document classification. In Proceedings of the 2016 conference 
##     of the North American chapter of the association for computational linguistics: human 
##     language technologies (pp. 1480-1489).
##
## Written by: Jasper Ginn <j.h.ginn@uu.nl>
## Course: Pattern Recognition @ Utrecht University

#%% Setup
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
import numpy as np
import os
from keras.preprocessing.text import Tokenizer # Use keras for tokenization & preprocessing
from keras import preprocessing
import matplotlib.pyplot as plt
from model_utils import load_FT, Embedding_FastText, WikiData, split, batcher, train_model

#%%

# Details for this script
from argparse import Namespace

# Model settings
args = Namespace(
  # File to save results
  out_file = 'results/basicNN_trials.csv',
  # Number of times to evaluate bayesian search for hyperparams
  max_evals = 200,
  # Size of the vocabulary
  input_vocabulary_size = 15000,
  # Embedding size
  embedding_dim = 300,
  # Max length of text sequences
  seq_max_len = 150,
  # NN settings
  learning_rate = 0.0001,
  batch_size = 128,
  epochs = 15,
  embedding_trainable = False
)

#%% Load pre-processed data
with open("data/WikiEssentials_L4_P3_preprocessed.pickle", "rb") as inFile:
    input_data = pickle.load(inFile)
    
# Process each
train_x = []
train_y = []
catmap = {}

# For each
for idx, itms in enumerate(input_data.items()):
  # Label and texts
  cat = idx
  txts = itms[1]
  catmap[cat] = itms[0]
  # For each text, append
  for doc, txt_lst in txts.items():
    xo= 0
    #if len(txt_lst) < 3:
    #  continue
    par_out = []
    for txt in txt_lst:
      if xo == 8:
        xo = 0
        break
      par_out.append(txt)
      xo += 1
    train_x.append(" ".join(par_out).replace("'s", ""))
    train_y.append(cat)

# Preprocess outcome label
train_y_ohe = np.zeros((len(train_y), len(input_data.keys())))
for idx,lbl in enumerate(train_y):
  train_y_ohe[idx, lbl] = 1

# Create tokenizer
tokenizer = Tokenizer(num_words=args.input_vocabulary_size,
                      lower = False,
                      filters = '!"$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')

# Fit on the documents
tokenizer.fit_on_texts(train_x)

#%% Number of unique words
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#%% To sequence (vectorize)
x_train = tokenizer.texts_to_sequences(train_x)

#%% Average length
seq_len = [len(x) for x in x_train]
print(np.median(seq_len))
print(np.max(seq_len))

# Pad sequences
train = preprocessing.sequence.pad_sequences(x_train, maxlen=args.seq_max_len)

# Get tokens to be looked up in FT embedding
WI = {k:v for k,v in tokenizer.word_index.items() if v <= (args.input_vocabulary_size - 1)}
FTEMB = load_FT("embeddings/wiki-news-300d-1M.vec", WI, args.embedding_dim, args.input_vocabulary_size)
# Check which are 0
io = np.sum(FTEMB, axis=1)
zerovar = np.where(io == 0)[0]
# Get words
zerovar_words = {k:v for k,v in WI.items() if v in zerovar}
zerovar_words

#%%

# Load data in Pytorch 'Dataset' format
# See 'model_utils.py'
VitalArticles = WikiData(train, train_y_ohe)

# Split data
# Returns two instances of 'WikiData' (train and test)
trainx, test = split(VitalArticles, val_prop = .05, seed = 856)

# Class weights
# These weights are unnormalized but that's what pytorch is expecting
cw = torch.tensor(np.max(np.sum(train_y_ohe, axis=0)) / (np.sum(train_y_ohe, axis=0))).type(torch.float).to(device)

#%% Attention layers

class Attention(nn.Module):
    def __init__(self, hidden_size):
        """
        Attention mechanism.

        :param hidden_size: size of the hidden states of the bidirectional GRU

        :seealso: https://pytorch.org/docs/stable/nn.html#gru for the output size of the GRU encoder (bidirectional)

        :return:
        """
        super(Attention, self).__init__()
        self._hidden_size = hidden_size
        # Linear layer for the tanh activation (eq. 5 in paper)
        #  (times two because bidirectional)
        self._layer1 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        # Linear layer for the softmax activation (eq. 6 in paper)
        self._layer2 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias = False)
    def forward(self, hidden_states):
        """
        Forward pass of the attention mechanism

        :param hidden_states: The hidden states of the input sequence at all times T.
        """
        # (see equation 5)
        u = F.tanh(self._layer1(hidden_states))
        # (see equation 6)
        alphas = F.softmax(self._layer2(u), dim=1)
        # --> current dimensions: X x Y x Z
        # Sentence vectors
        # (see equation 7)
        # Apply the attention weights (alphas) to each hidden state
        for idx in range(0, hidden_states.size(0)):
            # Get hidden state at time t
            hidden_current = hidden_states[idx]
            # Get attention weights at time t
            alphas_current = alphas[idx]
            # Hadamard product (element-wise)
            vector_weighted = hidden_current * alphas_current
            # Concatenate
            if idx > 0:
                s = torch.cat((s, vector_weighted), 0)
            else:
                s = vector_weighted
        # Sum across time axis (0)
        return(torch.sum(s, 0))

#%%

# Set up an embedding
embedding = Embedding_FastText(weights, freeze_layer = True)    
Encoder_GRU = nn.GRU(weights.shape[1], 32, bidirectional = True)

#%%

## Word-level GRU + attention
class HAN_word(nn.Module):
    def __init__(self, weights, hidden_dim):
        super(BaselineNN, self).__init__()
        # Get embedding dimensions
        self.weights_dim = weights.shape[1]
        # Set up embedding
        self.embedding = Embedding_FastText(weights, freeze_layer = True)
        # Set up bidirectional GRU encoder
        self.Encoder_GRU = nn.GRU(self.weights_dim, hidden_dim, bidirectional = True)
        # Attention layer
        self.attention
    def forward(self, input, dropout = 0):
        # Call embedding
        embedded = self.embedding(input).sum(dim=1)
        # Predict
        yhat = F.relu(self.linear1(embedded))
        # Dropout
        yhat = F.dropout(yhat, p = self._p_dropout)
        # Linear
        yhat = self.linear2(yhat)
        # Probabilities (logged)
        yhat = F.softmax(yhat, dim=1)
        return(yhat)

## Sentence-level GRU + attention


## Make HAN module