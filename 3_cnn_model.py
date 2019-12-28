## CNN model using pytorch

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
from keras import preprocessing # Sequence padding etc.
import matplotlib.pyplot as plt

# Details for this script
from argparse import Namespace

# Model settings
args = Namespace(
  # File to save results
  out_file = 'results/convNN_trials.csv',
  # Number of times to evaluate bayesian search for hyperparams
  max_evals = 500,
  # Size of the vocabulary
  input_vocabulary_size = 15000,
  # Embedding size
  embedding_dim = 300,
  # Max length of text sequences
  seq_max_len = 150,
  # NN settings
  learning_rate = 0.0008,
  batch_size = 128,
  epochs = 10,
  init_random_missing = False,
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

#%% To sequence
x_train = tokenizer.texts_to_sequences(train_x)

#%% Average length
seq_len = [len(x) for x in x_train]
print(np.median(seq_len))
print(np.max(seq_len))

# Pad sequences
train = preprocessing.sequence.pad_sequences(x_train, maxlen=args.seq_max_len)

# Load embeddings
from model_utils import load_FT, Embedding_FastText, WikiData, split, batcher, make_train_state

# Get tokens to be looked up in FT embedding
WI = {k:v for k,v in tokenizer.word_index.items() if v <= (args.input_vocabulary_size - 1)}
FTEMB = load_FT("models/wiki-news-300d-1M.vec", WI, args.embedding_dim, args.input_vocabulary_size)
# Check which are 0
io = np.sum(FTEMB, axis=1)
zerovar = np.where(io == 0)[0]
# Get words
zerovar_words = {k:v for k,v in WI.items() if v in zerovar}
zerovar_words

# Load pre-processed embeddings
with open("embeddings/prep.pickle", "rb") as inFile:
    FTEMB = pickle.load(inFile)

# Set up a convolutional network
class Convolutions(nn.Module):
    def __init__(self, weights, num_classes, max_seq_len, conv_filters = 100, filter_size = (1,2,3), batch_norm = True, p_dropout = 0):
        super(Convolutions, self).__init__()
        # Get embedding dimensions
        self.weights_dim = weights.shape[1]
        # Set up embedding
        self.embedding = Embedding_FastText(weights, freeze_layer = True)
        # Set batch norm
        self.batch_norm = batch_norm
        self.p_dropout = p_dropout
        # Set up convolutions
        self.convlayer1 = nn.Conv1d(in_channels=self.weights_dim,
                                                out_channels=conv_filters,
                                                kernel_size=filter_size[0])
        self.convlayer2 = nn.Conv1d(in_channels=self.weights_dim,
                                                out_channels=conv_filters,
                                                kernel_size=filter_size[1])
        self.convlayer3 = nn.Conv1d(in_channels=self.weights_dim,
                                                out_channels=conv_filters,
                                                kernel_size=filter_size[2])
        # Batch norm
        self.bn = nn.BatchNorm1d(num_features=3*conv_filters)
        # Softmax layer
        self.linear2 = nn.Linear(3*conv_filters, num_classes)
    def forward(self, input):
        # Call embedding
        embedded = self.embedding(input)
        # Permute
        #  --> we have: (batch_size, sentence_length, embedding_dim)
        #  --> we need: (batch_size, embedding_dim, sentence_length)
        embedded = embedded.permute(0, 2, 1)
        # Dropout
        embedded = F.dropout(embedded, 0.5)
        # Convolutions with ELU activation
        conv1_out = F.elu(self.convlayer1(embedded))
        conv2_out = F.elu(self.convlayer2(embedded))
        conv3_out = F.elu(self.convlayer3(embedded))
        # Global max pool across filters
        conv1_out, _ = conv1_out.max(dim=2)
        conv2_out, _ = conv2_out.max(dim=2)
        conv3_out, _ = conv3_out.max(dim=2)
        # Concatenate
        concat = torch.cat((conv1_out, conv2_out, conv3_out), 1)
        # Apply batch norm
        if self.batch_norm:
          concat = self.bn(concat)
        # Dropout
        pred = F.dropout(concat, p = self.p_dropout)
        # Linear
        yhat = self.linear2(pred)
        # Probabilities (logged)
        yhat = F.softmax(yhat, dim=1)
        return(yhat)

# Load data in Pytorch 'Dataset' format
VitalArticles = WikiData(train, train_y_ohe)

# Split data
trainx, test = split(VitalArticles, val_prop = .05, seed = 344)

# Class weights
cw = torch.tensor(np.round(np.max(np.sum(train_y_ohe, axis=0)) / (np.sum(train_y_ohe, axis=0)), 1)).type(torch.float).to(device)
# Set up the classifier
#conv = Convolutions(FTEMB, train_y_ohe.shape[1], args.seq_max_len, 100, (3,4,5), True, 0.3)
#conv = conv.to(device)  
#loss_function = nn.CrossEntropyLoss(weight=cw)  
# Optimizer
#optimizer = optim.Adam(conv.parameters(), lr=args.learning_rate)
#_, io = train_model(conv, trainx, optimizer, loss_function, 25, 0.1, 128, device = device)
  
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
def convNN_search(parameters):
  """Set up, run and evaluate a baseline neural network"""
  mlr = conv = Convolutions(FTEMB, train_y_ohe.shape[1], args.seq_max_len, 
                            parameters["filters"], (parameters["filter_size_1"], parameters["filter_size_2"], 
                            parameters["filter_size_3"]), parameters["batch_norm"], parameters["dropout"])
  # To device
  mlr = mlr.to(device)  
  loss_function = nn.CrossEntropyLoss(weight=cw)  
  # Optimizer
  if parameters["optimizer"] == "Adam":
    optimizer = optim.Adam(mlr.parameters(), lr=parameters["learning_rate"])
  else:
    optimizer = optim.RMSprop(mlr.parameters(), lr=parameters["learning_rate"])
  # Train using CV
  _, io = train_model(mlr, trainx, optimizer, loss_function, 25, 0.1, 128, device = device)
  # Write
  is_min = np.argmin(io["val_loss"])
  with open(args.out_file, 'a') as of_connection:
    writer = csv.writer(of_connection)
    writer.writerow([io["val_loss"][is_min], parameters, is_min, io["val_acc"][is_min]])
  # Return cross-validation loss
  return({"loss": np.min(io["val_loss"]), "parameters": parameters, "iteration": is_min, 'status':STATUS_OK})

# Test if works  
parameters = {"learning_rate": 0.00104, "filters": 100, "filter_size_1":3, 
              "filter_size_2":4, "filter_size_3":5, "batch_norm": True, 
              "dropout": 0.016, "optimizer": "Adam"}
po = convNN_search(parameters)

# Define the search space
space = {
    'filters': hp.choice('filters', [100, 150, 200, 250]),
    'optimizer': hp.choice("optimizer", ["Adam"]),
    'dropout': hp.uniform("dropout", 0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'batch_norm': hp.choice('batch_norm', [True, False]),
    'filter_size_1': hp.uniformint('filter_size_1', 1, 15),
    'filter_size_2': hp.uniformint('filter_size_2', 1, 15),
    'filter_size_3': hp.uniformint('filter_size_3', 1, 15)
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
best = fmin(fn = convNN_search, space = space, algo = tpe.suggest, 
            max_evals = args.max_evals, trials = bayes_trials)
            
# Save
import json
with open("hyperparameters/best_conv.json", "w") as outFile:
  # Cast to serializable files
  best["dropout"] = float(best["dropout"])
  best["hidden_units"] = 256
  best["learning_rate"] = float(best["learning_rate"])
  best["optimizer"] = "Adam"
  json.dump(best, outFile)
  
# Load
with open("hyperparameters/best_conv.json", "r") as inFile:
  parameters = json.load(inFile)

# To device
mlr = BaselineNN(FTEMB, train_y_ohe.shape[1], parameters["hidden_units"], p_dropout = parameters["dropout"])
mlr = mlr.to(device)  
loss_function = nn.CrossEntropyLoss(weight=cw)  
# Optimizer
optimizer = optim.Adam(mlr.parameters(), lr=parameters["learning_rate"])
mlr, io = train_model(mlr, trainx, optimizer, loss_function, 25, 0.1, 128, device = device)

mlr.eval()
# Predict
y_pred = mlr(torch.tensor(test.X).type(torch.long).to(device))
# Retrieve true y
val, y_true = torch.tensor(test.y).type(torch.long).to(device).max(dim=1)
# Loss
loss_val = loss_function(y_pred, y_true)
# Accuracy
_, y_pred = y_pred.max(dim=1)
acc_val = np.int((y_pred == y_true).sum()) / y_pred.size()[0]
acc_val
