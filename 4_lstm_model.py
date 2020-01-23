#%% This script preprocesses the wikipedia data and trains the LSTM Network.
# The LSTM Network utility functions can be found in LSTM.py

from LSTM import LSTMN, WikiDocData, split_data, train_lstmn, batcher, process_batch
import pickle
import numpy as np
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import optim
import torch.nn as nn
from sklearn import metrics

# Import namespace
from argparse import Namespace
from keras.preprocessing.text import Tokenizer # Use keras for tokenization & preprocessing
import itertools

# Helper functions
from preprocess_utils import tokenize_text

# Base directory
bd = "data/HAN"
# Create a map for the different sentence lengths
slmap = {}
for sl in ["S8", "S10", "S12", "S15"]:
    idx=int(sl.strip("S"))
    slmap[idx] = {"tokenizer": os.path.join(bd, "tokenizer_{}.pickle".format(sl)),
                 "embedding": os.path.join(bd, "HAN_embeddings_{}.pickle".format(sl)),
                 "input_data": os.path.join(bd, "data_{}.pickle".format(sl))}

# Model settings
args = Namespace(
    # Tokenizer
    data_files_map=slmap,
    # File to save results
    out_file='results/LSTM_trials.csv',
    # Number of times to evaluate bayesian search for hyperparams
    # NB: the HAN is very expensive to run even on a GPU ##########################
    max_evals=300,
    # Embedding size
    embedding_dim=300,
    # NN settings
    embedding_trainable=False,
    # Batch size
    batch_size=128
)

#%% Load the data for one of the sentence lengths

sent_length = 15

# Load
with open(args.data_files_map[sent_length]["tokenizer"], "rb") as inFile:
    tokenizer = pickle.load(inFile)
with open(args.data_files_map[sent_length]["embedding"], "rb") as inFile:
    FTEMB = torch.tensor(pickle.load(inFile)).to(device)
with open(args.data_files_map[sent_length]["input_data"], "rb") as inFile:
    data = pickle.load(inFile)
# Unpack
train_x, train_y = data["train_x"], data["train_y"]
docs_vectorized_lstm = []
doc_lstm = []
for doc in train_x:
    doc_lstm = []
    for sent in doc:
        doc_lstm = doc_lstm + sent
    docs_vectorized_lstm.append(doc_lstm)
train_x = docs_vectorized_lstm
labels_vect = data["labels_vectorized"]
idx_to_label = data["idx_to_label"]
label_to_idx = data["labels_to_idx"]

#%% View the max length of all sentences in all documents

# Max length of the documents
max_seq_doc = max([len(doc) for doc in train_x])

# View
print((max_seq_doc))

#%% Class weights (these are the same for each)

# Class weights
# Preprocess outcome label
train_y_ohe = np.zeros((len(train_y), len(data["labels_to_idx"])))
for idx,lbl in enumerate(train_y):
  train_y_ohe[idx, lbl] = 1
# These weights are unnormalized but that's what pytorch is expecting
cw = torch.tensor(np.max(np.sum(train_y_ohe, axis=0)) / (np.sum(train_y_ohe, axis=0))).type(torch.float).to(device)

#%% Unique classes

num_classes = len(np.unique(labels_vect))

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

# Function that sets up model and outputs and returns validation loss
def LSTMN_search(parameters):
    """Set up, run and evaluate a LSTMN"""
    # Based on the parameters, load various settings
    sent_length = parameters["sent_length"]
    # Load data
    with open(args.data_files_map[sent_length]["tokenizer"], "rb") as inFile:
        tokenizer = pickle.load(inFile)
    with open(args.data_files_map[sent_length]["embedding"], "rb") as inFile:
        FTEMB = torch.tensor(pickle.load(inFile)).to(device)
    with open(args.data_files_map[sent_length]["input_data"], "rb") as inFile:
        data = pickle.load(inFile)
    # Unpack
    train_x, train_y = data["train_x"], data["train_y"]
    docs_vectorized_lstm = []
    doc_lstm = []
    for doc in train_x:
        doc_lstm = []
        for sent in doc:
            doc_lstm = doc_lstm + sent
        docs_vectorized_lstm.append(doc_lstm)
    train_x = docs_vectorized_lstm
    labels_vect = data["labels_vectorized"]
    idx_to_label = data["idx_to_label"]
    label_to_idx = data["labels_to_idx"]
    # Set up the model
    WikiLSTM = LSTMN(FTEMB,
                  args.batch_size,
                  num_classes,
                  bidirectional = parameters["bidirectional"],
                  nb_lstm_layers = parameters["nb_lstm_layers"],
                  nb_lstm_units = parameters["nb_lstm_units"],
                  dropout_prop=parameters["dropout_prop"])
    
    # To cuda
    WikiLSTM.to(device)
    # Set up optimizer
    optimizer = optim.Adam(WikiLSTM.parameters(), lr=parameters["learning_rate"])
    # Criterion
    if parameters["use_class_weights"]:
      criterion = nn.CrossEntropyLoss(weight=cw)
    else:
      criterion = nn.CrossEntropyLoss()
    # Run the model
    WikiLSTM_out, history = train_lstmn(train_x, train_y, WikiLSTM, optimizer, criterion,
                                   epochs=10, val_split=0.1, batch_size=args.batch_size,
                                   device=device)
    # Max accuracy
    which_min = int(np.argmin(history["validation_loss"]))
    # Write to file
    with open(args.out_file, 'a') as of_connection:
        writer = csv.writer(of_connection)
        writer.writerow([parameters,
                         which_min,
                         np.round(history["training_loss"][which_min], 4),
                         np.round(history["validation_accuracy"][which_min], 4),
                         np.round(history["validation_loss"][which_min], 4),
                         np.round(history["validation_f1"][which_min], 4),
                         np.round(history["validation_precision"][which_min], 4),
                         np.round(history["validation_recall"][which_min], 4)])
    # Return cross-validation loss
    # NB: we are minimizing here zo we need to take 1-accuracy
    return({"loss": history["validation_loss"][which_min], "parameters": parameters, "iteration": which_min, 'status':STATUS_OK})

# Define the search space
space = {
    'nb_lstm_units': hp.choice('nb_lstm_units', [32,64,128]),
    'nb_lstm_layers': hp.choice('nb_lstm_layers', [1,2]),
    'bidirectional': hp.choice('bidirectional', [False, True]),
    'sent_length': hp.choice("sent_length", [8, 10, 12, 15]),
    'use_class_weights': hp.choice("use_class_weights", [True, False]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.03)),
    'dropout_prop': hp.uniform("dropout", 0, 0.5)
}

#%% Test space

# Test if works
from hyperopt.pyll.stochastic import sample
parameters = sample(space)
print(parameters)
#po = LSTMN_search(parameters)

#%% Run the optimizer

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
import datetime
begin = datetime.datetime.now()

best = fmin(fn = LSTMN_search, space = space, algo = tpe.suggest,
            max_evals = args.max_evals, trials = bayes_trials)

finish = datetime.datetime.now()
t_spent = begin-finish
print("began at: " + begin.strftime("%Y-%m-%d %H:%M"))
print("ended at: " + finish.strftime("%Y-%m-%d %H:%M"))
print("total time: " + str(t_spent))

#%% Train LSTMN on best parameters and train data

max_sent_length = 15

# Load data for the sentence length
sent_length = max_sent_length

with open("data/HAN/tokenizer_S{}.pickle".format(sent_length), "rb") as inFile:
    tokenizer = pickle.load(inFile)
with open("data/HAN/HAN_embeddings_S{}.pickle".format(sent_length), "rb") as inFile:
    FTEMB = torch.tensor(pickle.load(inFile)).to(device)
with open("data/HAN/data_S{}.pickle".format(sent_length), "rb") as inFile:
    data = pickle.load(inFile)
# Unpack
docs_vectorized = data["docs_vectorized"]
labels_vect = data["labels_vectorized"]
idx_to_label = data["idx_to_label"]
label_to_idx = data["labels_to_idx"]

#%%

best = { ###########################################################################################################
    "bidirectional" : True,
    "dropout_prop" : 0.17665639259474208,
    "learning_rate" : 0.02318120952027118,
    "nb_lstm_units" : 64,
    "nb_lstm_layers" : 1,
    "use_class_weights" : True,
    "batch_size" : 128,
    "num_classes" : len(np.unique(labels_vect)),
    "epochs" : 9
}

# Split
train, val = split_data(docs_vectorized_lstm, labels_vect, 6754, p=0.05)
# Make dataset
test = WikiDocData(val[0], val[1])
# Set up the model
WikiLSTM = LSTMN(FTEMB, best["batch_size"], best["num_classes"],
                 best["bidirectional"], best["nb_lstm_layers"],
                 best["nb_lstm_units"], best["dropout_prop"])
# To cuda
WikiLSTM.to(device)
# Set up optimizer
optimizer = optim.Adam(WikiLSTM.parameters(), lr= best["learning_rate"])
# Criterion
if best["use_class_weights"]:
    criterion = nn.CrossEntropyLoss(weight=cw)
else:
    criterion = nn.CrossEntropyLoss()

# Training routine
WikiLSTM_out, history = train_lstmn(train[0], train[1], WikiLSTM, optimizer, criterion,
                                epochs = best["epochs"], val_split = 0.1, batch_size = best["batch_size"],
                                device = device)

#%% Evaluate the model on test data

# For now, just make a single batch of the test data for evaluation
valbatch = batcher(test, len(test.X))

# Preprocess
seqs, lens = process_batch(valbatch, device = device)

# Predict
with torch.no_grad():
    WikiLSTM_out.eval()
    probs = WikiLSTM_out(seqs, lens)

# %% Classes

# To classes
out = torch.argmax(probs, dim=1).cpu().numpy()

# Get true label
ytrue = [batch[1] for batch in valbatch]
ytrue = torch.tensor(ytrue).cpu().numpy()

# Accuracy
print(sum(out == ytrue)/len(out))

#%% Print classification report

# Print classification report
print(metrics.classification_report(ytrue, out, target_names = list(label_to_idx.keys())))

#%% Save model

torch.save(WikiLSTM_out.state_dict(), "models/LSTM.pt")