#%% Baseline model

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
    # Tokenizer
    tokenizer="data/other/tokenizer.pickle",
    # Embedding
    embedding="data/other/embedding_matrix.pickle",
    # Input data
    input_data="data/other/vectorized_input_data.pickle",
    # File to save results
    out_file='results/basicNN_trials.csv',
    # Number of times to evaluate bayesian search for hyperparams
    max_evals=500,
    # Embedding size
    embedding_dim=300,
    # Max length of text sequences
    seq_max_len=150,
    # NN settings
    embedding_trainable = False
)

#%% Load pre-processed data

# Tokenizer
with open(args.tokenizer, "rb") as inFile:
    tokenizer = pickle.load(inFile)

# Embedding
with open(args.embedding, "rb") as inFile:
    FTEMB = pickle.load(inFile)

# Data
with open(args.input_data, "rb") as inFile:
    input_data = pickle.load(inFile)

# Unroll data
train_x=input_data["train_x"]
train_y=input_data["train_y"]
category_map=input_data["catmap"]

# Shuffle data
X, y = train_x, np.array(train_y)
np.random.seed(4352)
rp = np.random.permutation(y.shape[0])
X = X[rp,:]
y = y[rp]

# Also for test
Xt, yt = input_data["test_x"], np.array(input_data["test_y"])
np.random.seed(6666)
rpt = np.random.permutation(yt.shape[0])
Xt = Xt[rpt,:]
yt = yt[rpt]

# To wikidata class
WD = WikiData(train_x, train_y)
test = WikiData(input_data["test_x"], input_data["test_y"])

#%% One-layer NN with softmax on top

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

# Class weights
# Preprocess outcome label
train_y_ohe = np.zeros((len(train_y), len(category_map)))
for idx,lbl in enumerate(train_y):
  train_y_ohe[idx, lbl] = 1
# These weights are unnormalized but that's what pytorch is expecting
cw = torch.tensor(np.max(np.sum(train_y_ohe, axis=0)) / (np.sum(train_y_ohe, axis=0))).type(torch.float).to(device)

#%% Callback functions

from sklearn import metrics
def f1_score(net, X, y):
    """Compute the F1 score"""
    ypred = net.predict(X)
    out_class = np.argmax(ypred, axis=1)
    return(metrics.f1_score(y, out_class, average="weighted"))

def precision_score(net, X, y):
    """Compute precision"""
    ypred = net.predict(X)
    out_class = np.argmax(ypred, axis=1)
    return(metrics.precision_score(y, out_class, average="weighted"))

def recall_score(net, X, y):
    """Compute recall"""
    ypred = net.predict(X)
    out_class = np.argmax(ypred, axis=1)
    return(metrics.recall_score(y, out_class, average="weighted"))

def accuracy_score(net, X, y):
    """Compute accuracy"""
    ypred = net.predict(X)
    out_class = np.argmax(ypred, axis=1)
    return(metrics.accuracy_score(y, out_class))

#%% Use hyperopt (Bayesian hyperparameter optimization) to search for good hyperparams

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
import skorch

# Function that sets up model and outputs and returns validation loss
def baselineNN_search(parameters):
  """Set up, run and evaluate a baseline neural network"""
  # CV with skorch
  net = NeuralNet(
    # Module
    module=BaselineNN,
    # Module settings
    module__hidden_dim = parameters["hidden_units"],
    module__p_dropout = parameters["dropout"],
    module__use_batch_norm = parameters["use_batch_norm"],
    module__weights = FTEMB, # These are word embeddings
    module__num_classes = len(category_map),
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
    device = device,
    # Callbacks
    callbacks=[
      skorch.callbacks.EpochScoring(f1_score, use_caching=True, name="valid_f1"),
      skorch.callbacks.EpochScoring(precision_score, use_caching=True, name="valid_precision"),
      skorch.callbacks.EpochScoring(recall_score, use_caching=True, name="valid_recall"),
      skorch.callbacks.EpochScoring(accuracy_score, use_caching=True, name="valid_accuracy")
    ]
  )
  # Verbose to false
  net.verbose = 1
  # Fit
  net = net.fit(WD)
  # Get train / validation history
  train_loss = net.history[:,"train_loss"]
  val_loss = net.history[:, "valid_loss"]
  val_accuracy = net.history[:, "valid_accuracy"]
  val_f1 = net.history[:,"valid_f1"]
  val_precision = net.history[:,"valid_precision"]
  val_recall = net.history[:,"valid_recall"]
  # Min loss
  which_min = np.argmin(val_loss)
  # Write to file
  with open(args.out_file, 'a') as of_connection:
    writer = csv.writer(of_connection)
    writer.writerow([parameters,
                     which_min,
                     np.round(train_loss[which_min], 4),
                     np.round(val_accuracy[which_min], 4),
                     np.round(val_loss[which_min], 4),
                     np.round(val_f1[which_min], 4),
                     np.round(val_precision[which_min], 4),
                     np.round(val_recall[which_min], 4)])
  # Return cross-validation loss
  return({"loss": val_loss[which_min], "parameters": parameters, "iteration": which_min, 'status':STATUS_OK})

# Define the search space
space = {
    'hidden_units': hp.choice('hidden_units', [64,128,256,512]),
    'optimizer': hp.choice("optimizer", ["Adam", "RMSprop"]),
    'use_batch_norm': hp.choice("use_batch_norm", [True, False]),
    'dropout': hp.uniform("dropout", 0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.02))
}

# Test if works
from hyperopt.pyll.stochastic import sample
params = sample(space)
po = baselineNN_search(params)

#%% Run bayesian optimization

# Algorithm
tpe_algorithm = tpe.suggest

# Trials object to track progress
bayes_trials = Trials()
# File to save first results
with open(args.out_file, 'w') as of_connection:
  writer = csv.writer(of_connection)
  # Write the headers to the file
  writer.writerow(['params',
                   'iteration',
                   'train_loss',
                   'val_accuracy',
                   'val_loss',
                   "val_f1",
                   "val_precision",
                   "val_recall"])

# Optimize
best = fmin(fn = baselineNN_search, space = space, algo = tpe.suggest,
            max_evals = args.max_evals, trials = bayes_trials)

#%% Fit on train & test

import skorch
from sklearn import metrics
from skorch import NeuralNet
from skorch.dataset import CVSplit

# Best parameters
best = Namespace(
    hidden_units=512,
    dropout=0.01285,
    learning_rate=0.001435,
    optimizer=optim.Adam,
    use_batch_norm=True,
    iterations=23,
    batch_size=128
)

# Run the model with the best parameters
net = NeuralNet(
    # Module
    module=BaselineNN,
    # Module settings
    module__hidden_dim = best.hidden_units,
    module__p_dropout = best.dropout,
    module__use_batch_norm = best.use_batch_norm,
    module__weights = FTEMB,
    module__num_classes = len(category_map),
    # Epochs & learning rate
    max_epochs=best.iterations,
    lr=best.learning_rate,
    # Optimizer
    optimizer=best.optimizer,
    # Loss function
    criterion=nn.CrossEntropyLoss,
    criterion__weight = cw,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    # Batch size
    batch_size = best.batch_size,
    train_split = CVSplit(cv=5),
    # Device
    device = device,
    # Callbacks
    callbacks=[
        skorch.callbacks.EpochScoring(f1_score, use_caching=True, name="valid_f1"),
        skorch.callbacks.EpochScoring(precision_score, use_caching=True, name="valid_precision"),
        skorch.callbacks.EpochScoring(recall_score, use_caching=True, name="valid_recall"),
        skorch.callbacks.EpochScoring(accuracy_score, use_caching=True, name="valid_accuracy")
    ]
)

# Verbose to false
net.verbose = 1

#%% Fit the model

io = net.fit(WD)

# Save model
net.save_params(f_params='models/baselineNN.pkl')

#%% Or load it from disk

net.initialize()
net.load_params(f_params="models/baselineNN.pkl")

#%% Predict on train

# Out
yhat = net.predict(WD)
# Classes
yhatc = yhat.argmax(axis=1)
# True labels
ytrue = WD.y
(ytrue == yhatc).sum() / yhatc.size

# Classification report
from sklearn import metrics
print(metrics.classification_report(ytrue, yhatc, target_names=list(category_map.values())))
metrics.confusion_matrix(ytrue, yhatc)

#%% Predict on test

# Out
yhat = net.predict(test)
# Classes
yhatc = yhat.argmax(axis=1)
# True labels
ytrue = test.y
(ytrue == yhatc).sum() / yhatc.size

# Classification report
from sklearn import metrics
print(metrics.classification_report(ytrue, yhatc, target_names=list(category_map.values())))
metrics.confusion_matrix(ytrue, yhatc)

#%% Save predictions

import pandas as pd
out_preds = pd.DataFrame({"yhat": yhatc, "ytrue":ytrue})
# Save
out_preds.to_csv("predictions/baseline.csv", index=False)
