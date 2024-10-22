#%% This script preprocesses the wikipedia data and trains the HAN.
# The HAN utility functions can be found in HAN.py

from HAN import HAN, WikiDocData, split_data, train_han, batcher, process_batch, predict_HAN
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
import os
import pickle

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
    out_file='results/HAN_trials.csv',
    # Number of times to evaluate bayesian search for hyperparams
    # NB: the HAN is very expensive to run even on a GPU
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
labels_vect = data["labels_vectorized"]
idx_to_label = data["idx_to_label"]
label_to_idx = data["labels_to_idx"]

#%% View the max length of all sentences in all documents

# Max length of the sentences
# (itertools.chain(*X)) makes list of lists into one, flat list
max_seq_len = max([len(seq) for seq in itertools.chain(*train_x)])

# Max length of documents (shoudl all be the same)
max_seq_doc = max([len(doc) for doc in train_x])

# View
print((max_seq_len))
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
# Tracking precision//recall//F1
from sklearn import metrics

# Function that sets up model and outputs and returns validation loss
def HAN_search(parameters):
    """Set up, run and evaluate a HAN"""
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
    labels_vect = data["labels_vectorized"]
    idx_to_label = data["idx_to_label"]
    label_to_idx = data["labels_to_idx"]
    # Set up the model
    WikiHAN = HAN(FTEMB,
                  parameters["hidden_size"],
                  parameters["hidden_size"],
                  args.batch_size,
                  num_classes,
                  dropout_prop=parameters["dropout_prop"])
    # To cuda
    WikiHAN.to(device)
    # Set up optimizer
    optimizer = optim.Adam(WikiHAN.parameters(), lr=parameters["learning_rate"])
    # Criterion
    if parameters["use_class_weights"]:
      criterion = nn.CrossEntropyLoss(weight=cw)
    else:
      criterion = nn.CrossEntropyLoss()
    # Run the model
    WikiHAN_out, history = train_han(train_x, train_y, WikiHAN, optimizer, criterion,
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
    'hidden_size': hp.choice('hidden_units', [32,64,128]),
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
po = HAN_search(parameters)

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
best = fmin(fn = HAN_search, space = space, algo = tpe.suggest,
            max_evals = args.max_evals, trials = bayes_trials)

#%% Train HAN on best parameters and train data

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
labels_vect = data["labels_vectorized"]
idx_to_label = data["idx_to_label"]
label_to_idx = data["labels_to_idx"]

# Best parameters
best = Namespace(
    dropout_prop = 0.165,
    hidden_size = 64,
    use_class_weights = True,
    batch_size = 128,
    num_classes = len(np.unique(labels_vect)),
    learning_rate = 0.007472,
    epochs = 9
)

# Set up the model
WikiHAN = HAN(FTEMB, best.hidden_size, best.hidden_size, best.batch_size, best.num_classes, dropout_prop=best.dropout_prop)
# To cuda
WikiHAN.to(device)
# Set up optimizer
optimizer = optim.Adam(WikiHAN.parameters(), lr= best.learning_rate)
# Criterion
if best.use_class_weights:
    criterion = nn.CrossEntropyLoss(weight=cw)
else:
    criterion = nn.CrossEntropyLoss()

#%% Training routine
WikiHAN_out, history = train_han(train_x, train_y, WikiHAN, optimizer, criterion,
                                epochs = best.epochs, val_split = 0.1, batch_size = best.batch_size,
                                device = device)

# Save model

torch.save(WikiHAN_out.state_dict(), "models/HAN.pt")

#%% Or load from existing model

WikiHAN.load_state_dict(torch.load("models/HAN.pt", map_location=torch.device(device)))
#%%
WikiHAN_out = WikiHAN

#%% Make datasets to get accuracy etc.

test_x, test_y = data["test_x"], data["test_y"]

# To dataset
test = WikiDocData(test_x, test_y)
train= WikiDocData(train_x, train_y)

#%% Get predictions for train data

yhat, ytruth = predict_HAN(model=WikiHAN_out, dataset=train, batch_size=128, device = device)

# Print classification report
print(metrics.classification_report(ytruth, yhat, target_names = list(label_to_idx.keys())))

#%% Get predictions for test data

yhat, yprob, ytruth = predict_HAN(model=WikiHAN_out, dataset=test, batch_size=128, return_probabilities=True, device = device)

# Print classification report
print(metrics.classification_report(ytruth, yhat, target_names = list(label_to_idx.keys())))

#%% Confusion matrix

print(metrics.confusion_matrix(ytruth, yhat))


#%% Save predictions

import pandas as pd
# Save
out_preds = pd.DataFrame({"yhat": yhat, "ytrue":ytruth})
out_preds.to_csv("predictions/HAN.csv", index=False)

#%% Save probabilities

out_probs = pd.DataFrame(yprob)
out_probs.to_csv("predictions/HAN_probs.csv", index = False)
#%% Save model

torch.save(WikiHAN_out.state_dict(), "models/HAN.pt")
word_weights = [we.cpu() for we in word_weights]
sent_weights = [se.cpu() for se in sent_weights]
print(len(word_weights))
print(len(sent_weights))

# %%

# The word-level weights (one for each sentence) are of shape:
#  (batch_size, sentence_length, hidden_dim)
# We can get attention weights for each of the words as follows:
#  (1) Sum across the hidden states 
#       This tells us how much weight is placed on the word across hidden units
#  (2) Subset the resulting vector by the actual sentence length of each input
#       The sentences are padded by batch size length (i.e. the length of the longest sentence)
#       We then subset the weights for the actual sentence length
#  (3) Normalize the weights for each of the sentences
word_weights[0].shape

# Reverse word index so it is easy to go from vectorized word ==> actual word
idx_to_word = {v:k for k,v in tokenizer.word_index.items()}

def word_attention(attention_vector, seq, idx_to_word):
    """
    Compute attention weights for each word in the sentence
    
    :param attention_vector: tensor of shape (sentence_length, word_hidden_dim)
    :param seq: the vectorized sequence of words
    :param idx_to_word: dict that maps sequence integers to words
    
    :return: dictionary where keys are the words in the sequence and value is the attention weight
    """
    # Sequence length
    seq_len = seq.shape[0]
    # Sum across hidden dimension (last axis)
    attention_summed = attention_vector.sum(axis=-1)
    # Subset
    attention_summed = attention_summed[:seq_len]
    # Normalize
    attention_normed = list(np.round(attention_summed / np.sum(attention_summed), 4))
    # Store
    return({idx_to_word[int(seq[idx])]:attention_normed[idx] for idx in range(seq_len)})

def sentence_attention(attention_vector):
    """
    Compute attention weights for each sentence

    :param attention_vector: tensor of shape (examples, sentences, sentence_hidden_dim)

    :return: dictionary where keys are sentence indices and values are sentence attention weights
    """
    # Create weights for each sample
    sent_weight = attention_vector.sum(axis=-1)
    # Normalize
    sent_weight /= sent_weight.sum()
    # To dict & return
    return({k:np.round(float(list(sent_weight)[k]), 3) for k in range(0, sent_weight.shape[0])})

import html
from IPython.core.display import display, HTML

# Prevent special characters like & and < to cause the browser to display something other than what you intended.
# Taken from: https://adataanalyst.com/machine-learning/highlight-text-using-weights/
def html_escape(text):
    return html.escape(text)

def make_word_weights(attention_weights):
    """
    make colored word attention weights

    :param attention_weights: weights returned by 'weights_attention'

    :return: returns HTML which can be plotted using: plot_word_attention_weights()

    :seealso: function adapted from
        - https://adataanalyst.com/machine-learning/highlight-text-using-weights/
    """
    # Maximum highligh value
    max_alpha = 0.8 
    highlighted_text = []
    # For each word and weight, create the HTML
    for word, weight in attention_weights.items():
        if weight is not None:
            highlighted_text.append('<span style="background-color:rgba(135,206,250,' + str(weight / max_alpha) + ');">' + html_escape(word) + '</span>')
        else:
            highlighted_text.append(word)
    # Join HTML
    highlighted_text = ' '.join(highlighted_text)
    # return
    return(highlighted_text)

def plot_word_attention_weights(highlighted_text):
    """
    Given some output from 'make_word_weights()' function, plot highlighted text

    :param highlighted_text: output from 'make_word_weights()'

    :return: plots the highlighted text
    """
    display(HTML(highlighted_text))


def plot_normed_word_weights(word_attention_vectors, sentence_attention_vectors):
    """
    Plot a 
    """

# %%

doc_idx = 25
sentence_idx =7
# Subset attention vector
attv = word_weights[sentence_idx].numpy()
# valbatch[example][X || y][sentence]
seq = valbatch[doc_idx][0][sentence_idx].numpy()
# Compute attention weights
att_weights = word_attention(attv[doc_idx,:,:], seq, idx_to_word)
# Print output label
print(idx_to_label[int(valbatch[doc_idx][1].numpy())])
print(idx_to_label[int(out[doc_idx])])
print(att_weights)

# %% Plot weights

plot_word_attention_weights(make_word_weights(att_weights))

# %%

import itertools

doc_idx = 3
print(idx_to_label[int(valbatch[doc_idx][1].numpy())])
word_weights_by_sentence = []
word_weights_original = []
# Get sentence attention weights
sa = sentence_attention(sent_weights[9][doc_idx,:,:].numpy())
# Weight the word attention weights by the sentence weights
for sentence_idx in range(0, len(word_weights)):
    # Subset attention vector
    attv = word_weights[sentence_idx].numpy()
    # Get the vectorized sequence ('sentence')
    seq = valbatch[doc_idx][0][sentence_idx].numpy()
    # Compute the attention weights
    att_weights = word_attention(attv[doc_idx,:,:], seq, idx_to_word)
    word_weights_original.append(att_weights)
    # Weight by sentence weight
    att_weights = {k:v*sa[sentence_idx] for k,v in att_weights.items()}
    # Push
    word_weights_by_sentence.append(att_weights)

# Norm weights over all words in the document
sum_total = sum(itertools.chain(*[list(weights.values()) for weights in word_weights_by_sentence]))
normed_weights = []
for weight in word_weights_by_sentence:
    normed_weights.append({k:v/sum_total for k,v in weight.items()})

# %% Highlight text in each sentence and concatenate

ww = [make_word_weights(weight) for weight in normed_weights]
plot_word_attention_weights(".<br>".join(ww))

# %%

sa

# Idea: check for each class how important e.g. the first sentence is.

# %%


# %%
