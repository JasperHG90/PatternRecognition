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
from model_utils import load_FT, Embedding_FastText, WikiData, split

# Details for this script
from argparse import Namespace

# Model settings
args = Namespace(
  # File to save results
  out_file = 'results/basicNN_trials.csv',
  # Number of times to evaluate bayesian search for hyperparams
  max_evals = 500,
  # Size of the vocabulary
  input_vocabulary_size = 20000,
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
# Save embeddings
with open("embeddings/prep.pickle", 'wb') as handle:
  pickle.dump(FTEMB,  handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Load pre-processed embeddings
with open("embeddings/prep.pickle", "rb") as inFile:
    FTEMB = pickle.load(inFile)

# Set up a softmax layer
# One-layer NN with softmax on top
class BaselineNN(nn.Module):
    def __init__(self, weights, num_classes, hidden_dim, p_dropout = 0, use_batch_norm = True):
        super(BaselineNN, self).__init__()
        # Set dropout percentage
        self._p_dropout = p_dropout
        # Get embedding dimensions
        self.weights_dim = weights.shape[1]
        # Set up embedding
        self.embedding = Embedding_FastText(weights, freeze_layer = True)
        # Set up hidden layer
        self.linear1 = nn.Linear(self.weights_dim, hidden_dim)
        # Set up batch norm
        self.bn = nn.BatchNorm1d(num_features=hidden_dim)
        self.use_batch_norm = use_batch_norm
        # Set up softmax layer
        self.linear2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, input, dropout = 0):
        # Call embedding
        embedded = self.embedding(input).sum(dim=1)
        # Predict
        yhat = self.linear1(embedded)
        # Apply batch norm
        if self.use_batch_norm:
          yhat = self.bn(yhat)
        # Activation
        yhat = F.relu(yhat)
        # Dropout
        yhat = F.dropout(yhat, p = self._p_dropout)
        # Linear
        yhat = self.linear2(yhat)
        # Probabilities
        yhat = F.softmax(yhat, dim=1)
        return(yhat)

# Load data in Pytorch 'Dataset' format
# See 'model_utils.py'
VitalArticles = WikiData(train, np.array(train_y))

# Split data
# Returns two instances of 'WikiData' (train and test)
trainx, test = split(VitalArticles, val_prop = .05, seed = 856)

# Class weights
# Preprocess outcome label
train_y_ohe = np.zeros((len(train_y), len(input_data.keys())))
for idx,lbl in enumerate(train_y):
  train_y_ohe[idx, lbl] = 1
# These weights are unnormalized but that's what pytorch is expecting
cw = torch.tensor(np.max(np.sum(train_y_ohe, axis=0)) / (np.sum(train_y_ohe, axis=0))).type(torch.float).to(device)

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
# Use skorch for cross-validation
from skorch import NeuralNet
from skorch.dataset import CVSplit
# Tracking precision//recall//F1
from sklearn import metrics

# Function that sets up model and outputs and returns validation loss
def baselineNN_search(parameters):
  """Set up, run and evaluate a baseline neural network"""
  # Split into train/test set
  train_current, test_current = split(trainx, val_prop = .05)
  # CV with skorch
  net = NeuralNet(
    # Module
    module=BaselineNN,
    # Module settings
    module__hidden_dim = parameters["hidden_units"],
    module__p_dropout = parameters["dropout"],
    module__use_batch_norm = parameters["use_batch_norm"],
    module__weights = FTEMB, # These are word embeddings
    module__num_classes = len(catmap),
    # Epochs & learning rate
    max_epochs=25,
    lr=parameters["learning_rate"],
    # Optimizer
    optimizer=optim.Adam if parameters["optimizer"] == "Adam" else optim.RMSprop,
    # Loss function
    criterion=nn.CrossEntropyLoss,
    criterion__weight = cw,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    # Batch size
    batch_size = 128,
    train_split = CVSplit(cv=5),
    # Device
    device = device
  )
  # Verbose to false
  net.verbose = 0
  # Fit
  net = net.fit(train_current)
  # Get train / validation history
  train_loss = net.history[:,"train_loss"]
  val_loss = net.history[:, "valid_loss"]
  # Min loss
  which_min = np.argmin(val_loss)
  # Predict on hold-out set
  yhat = net.predict(test_current)
  yhatc = yhat.argmax(axis=1)
  ytrue = test_current.y
  # Get accuracy
  acc_test = np.round((ytrue == yhatc).sum() / yhatc.size, 4)
  # Prec/rec/f1
  out_metrics = metrics.precision_recall_fscore_support(ytrue,
                                                        yhatc,
                                                        average = "weighted")
  # Write to file
  with open(args.out_file, 'a') as of_connection:
    writer = csv.writer(of_connection)
    writer.writerow([parameters, np.round(train_loss[which_min], 4),
                     np.round(val_loss[which_min], 4), which_min,
                     np.round(acc_test, 4), np.round(out_metrics[0], 4),
                     np.round(out_metrics[1], 4), np.round(out_metrics[2], 4)])
  # Return cross-validation loss
  return({"loss": val_loss[which_min], "parameters": parameters, "iteration": which_min, 'status':STATUS_OK})

# Define the search space
space = {
    'hidden_units': hp.choice('hidden_units', [64,128,256,512]),
    'optimizer': hp.choice("optimizer", ["Adam", "RMSprop"]),
    'use_batch_norm': hp.choice("use_batch_norm", [True, False]),
    'dropout': hp.choice("dropout", [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.02))
}

# Test if works
from hyperopt.pyll.stochastic import sample
parameters = sample(space)
po = baselineNN_search(parameters)

# Algorithm
tpe_algorithm = tpe.suggest

# Trials object to track progress
bayes_trials = Trials()
# File to save first results
with open(args.out_file, 'w') as of_connection:
  writer = csv.writer(of_connection)
  # Write the headers to the file
  writer.writerow(['params', 'train_loss', 'val_loss', 'iteration', 'test_accuracy', "test_f1", "test_precision", "test_recall"])

# Optimize
best = fmin(fn = baselineNN_search, space = space, algo = tpe.suggest,
            max_evals = args.max_evals, trials = bayes_trials)

# Run the model with the best paramaters
net = NeuralNet(
    # Module
    module=BaselineNN,
    # Module settings
    module__hidden_dim = 512,
    module__p_dropout = 0.1,
    module__use_batch_norm = True,
    module__weights = FTEMB,
    module__num_classes = len(catmap),
    # Epochs & learning rate
    max_epochs=25,
    lr=0.00172,
    # Optimizer
    optimizer=optim.Adam,
    # Loss function
    criterion=nn.CrossEntropyLoss,
    criterion__weight = cw,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    # Batch size
    batch_size = 128,
    train_split = CVSplit(cv=5),
    # Device
    device = device
)

# Verbose to false
net.verbose = 0
# Fit
io = net.fit(trainx)
yhat = net.predict(test)
yhatc = yhat.argmax(axis=1)
ytrue = test.y
(ytrue == yhatc).sum() / yhatc.size

# Classification report
from sklearn import metrics
print(metrics.classification_report(ytrue, yhatc, target_names=list(catmap.values())))
metrics.confusion_matrix(ytrue, yhatc)
