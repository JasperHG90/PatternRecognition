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
from model_utils import load_FT, Embedding_FastText, WikiData, split, batcher, make_train_state

# Details for this script
from argparse import Namespace

# Model settings
args = Namespace(
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
FTEMB = load_FT("models/wiki-news-300d-1M.vec", WI, args.embedding_dim, args.input_vocabulary_size)
# Check which are 0
io = np.sum(FTEMB, axis=1)
zerovar = np.where(io == 0)[0]
# Get words
zerovar_words = {k:v for k,v in WI.items() if v in zerovar}
zerovar_words

# Set up a softmax layer
# One-layer NN with softmax on top
class Multinomial(nn.Module):
    def __init__(self, weights, num_classes, hidden_dim):
        super(Multinomial, self).__init__()
        # Get embedding dimensions
        self.weights_dim = weights.shape[1]
        # Set up embedding
        self.embedding = Embedding_FastText(weights, freeze_layer = True)
        # Set up hidden layer
        self.linear1 = nn.Linear(self.weights_dim, hidden_dim)
        # Set up softmax layer
        self.linear2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, input):
        # Call embedding
        embedded = self.embedding(input).sum(dim=1)
        # Predict
        yhat = F.relu(self.linear1(embedded))
        # Dropout
        #yhat = F.dropout(yhat, p = 0.3)
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
mlr = Multinomial(FTEMB, train_y_ohe.shape[1], 128)
mlr = mlr.to(device)  
loss_function = nn.CrossEntropyLoss(weight=cw)  
# Optimizer
optimizer = optim.Adam(mlr.parameters(), lr=args.learning_rate)
# Dictionary to store results
train_state = make_train_state(args)
  
# Epochs
for epoch_idx in range(args.epochs):
  # Split train / test
  trn, tst = split(trainx, val_prop=0.1)
  # Create training batches
  batches = batcher(trn, batch_size = args.batch_size, shuffle = True, device = "cpu")
  # Keep track of loss
  loss = 0.0
  acc = 0.0
  # Training mode
  mlr.train()
  # For each batch ...
  for batch_idx, batch_data in enumerate(batches):
    # Training loop
    # --------------------
    # Zero gradients
    optimizer.zero_grad()
    # Compute output
    probs = mlr(batch_data["X"].type(torch.long))
    # Classes
    valt, y = batch_data["y"].type(torch.long).max(axis=1)
    # Compute loss
    loss_batch = loss_function(probs, y)
    loss += (loss_batch.item() - loss) / (batch_idx + 1)
    # Compute gradients
    loss_batch.backward()
    # Gradient descent
    optimizer.step()
    #--------------------
    # End training loop
    # Compute accuracy
    val, yhat = probs.max(axis=1)
    acc_batch = np.int((yhat == y).sum()) / yhat.size()[0]
    acc += (acc_batch - acc) / (batch_idx + 1)
  # Add loss/acc
  train_state["train_loss"].append(np.round(loss, 4))
  train_state["train_acc"].append(np.round(acc, 4))
  # Predict on validation set
  mlr.eval()
  # Predict
  y_pred = mlr(torch.tensor(tst.X).type(torch.long))
  # Retrieve true y
  val, y_true = torch.tensor(tst.y).type(torch.long).max(axis=1)
  # Loss
  loss_val = loss_function(y_pred, y_true)
  # Accuracy
  _, y_pred = y_pred.max(axis=1)
  acc_val = np.int((y_pred == y_true).sum()) / y_pred.size()[0]
  # Add
  train_state["val_loss"].append(np.round(loss_val.item(), 4))
  train_state["val_acc"].append(np.round(acc_val, 4))
