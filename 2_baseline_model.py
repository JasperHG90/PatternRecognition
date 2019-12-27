## Baseline model

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

# Set up a softmax layer
# One-layer NN with softmax on top
class BaselineNN(nn.Module):
    def __init__(self, weights, num_classes, hidden_dim, p_dropout = 0):
        super(BaselineNN, self).__init__()
        self._p_dropout = p_dropout
        # Get embedding dimensions
        self.weights_dim = weights.shape[1]
        # Set up embedding
        self.embedding = Embedding_FastText(weights, freeze_layer = True)
        # Set up hidden layer
        self.linear1 = nn.Linear(self.weights_dim, hidden_dim)
        # Set up softmax layer
        self.linear2 = nn.Linear(hidden_dim, num_classes)
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

# Load data in Pytorch 'Dataset' format
# See 'model_utils.py'
VitalArticles = WikiData(train, train_y_ohe)

# Split data
# Returns two instances of 'WikiData' (train and test)
trainx, test = split(VitalArticles, val_prop = .05)

# Class weights
# These weights are unnormalized but that's what pytorch is expecting
cw = torch.tensor(np.max(np.sum(train_y_ohe, axis=0)) / (np.sum(train_y_ohe, axis=0))).type(torch.float).to(device)
# Set up the classifier
# Hidden dim neurons: 128
#mlr = BaselineNN(FTEMB, train_y_ohe.shape[1], 128)
#mlr = mlr.to(device)  
#loss_function = nn.CrossEntropyLoss(weight=cw)  
# Optimizer
#optimizer = optim.Adam(mlr.parameters(), lr=args.learning_rate)

### Use hyperopt (Bayesian hyperparameter optimization) to search for good hyperparams
from hyperopt import STATUS_OK
import csv
from hyperopt import hp
# Optimizer
from hyperopt import tpe
# Save basic training information
from hyperopt import Trials
# Optimizer criterion
from hyperopt import fmin

# Function that sets up model and outputs and returns validation loss
def baselineNN_search(parameters):
  """Set up, run and evaluate a baseline neural network"""
  mlr = BaselineNN(FTEMB, train_y_ohe.shape[1], parameters["hidden_units"], p_dropout = parameters["dropout"])
  # To device
  mlr = mlr.to(device)  
  loss_function = nn.CrossEntropyLoss(weight=cw)  
  # Optimizer
  optimizer = optim.Adam(mlr.parameters(), lr=parameters["learning_rate"])
  # Train using CV
  _, io = train_model(mlr, trainx, optimizer, 25, 0.1, 128)
  # Write
  is_min = np.argmin(io["val_loss"])
  with open(args.out_file, 'a') as of_connection:
    writer = csv.writer(of_connection)
    writer.writerow([io["val_loss"][is_min], parameters, is_min, io["val_acc"][is_min]])
  # Return cross-validation loss
  return({"loss": np.min(io["val_loss"]), "parameters": parameters, 'status':STATUS_OK})

# Test if works  
parameters = {"learning_rate": 0.001, "hidden_units": 128, "dropout": 0}
po = baselineNN_search(parameters)

# Define the search space
space = {
    'hidden_units': hp.choice('hidden_units', [32,64,128,256]),
    'dropout': hp.uniform("dropout", 0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01))
}

# Algorithm
tpe_algorithm = tpe.suggest

# Trials object to track progress
bayes_trials = Trials()
# File to save first results
with open(args.out_file, 'w') as of_connection:
  writer = csv.writer(of_connection)
  # Write the headers to the file
  writer.writerow(['loss', 'params', 'iteration', 'accuracy'])

# Optimize
best = fmin(fn = baselineNN_search, space = space, algo = tpe.suggest, 
            max_evals = args.max_evals, trials = bayes_trials)
