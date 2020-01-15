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
  max_evals = 300,
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
from model_utils import load_FT, Embedding_FastText, WikiData, split

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
    def __init__(self, weights, num_classes, max_seq_len, conv_filters = 100, conv_layers = 3, filter_size = (1,2,3), batch_norm = True, p_dropout = 0):
        super(Convolutions, self).__init__()
        # Get embedding dimensions
        self.weights_dim = weights.shape[1]
        # Set up embedding
        self.embedding = Embedding_FastText(weights, freeze_layer = True)
        # Set batch norm
        self.apply_batch_norm = batch_norm
        self.p_dropout = p_dropout
        self.number_layers = conv_layers
        # Layers must equal filters
        assert conv_layers == len(filter_size), "Number of layers must equal length of filter sizes"
        # Set up convolutions
        self.convlayers = nn.ModuleList()
        self.bn = nn.ModuleList()
        for fs in filter_size:
          self.convlayers.append(nn.Conv1d(in_channels=self.weights_dim,
                                            out_channels=conv_filters,
                                            kernel_size=fs))
          self.bn.append(nn.BatchNorm1d(num_features=conv_filters))
        # Softmax layer
        self.linear2 = nn.Linear(conv_layers*conv_filters, num_classes)
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
        # Global max pool across filters
        conv_out_mp = []
        for b,l in zip(self.bn, self.convlayers):
          co = l(embedded)
          # Batch norm
          if self.apply_batch_norm:
            co = b(co)
          # Relu
          co = F.relu(co)
          # Global max pool
          comp, _ = co.max(dim=2)
          # Add
          conv_out_mp.append(comp)
        # Concatenate
        concat = torch.cat(tuple(conv_out_mp), 1)
        # Dropout
        pred = F.dropout(concat, p = self.p_dropout)
        # Linear
        yhat = self.linear2(pred)
        # Probabilities (logged)
        yhat = F.softmax(yhat, dim=1)
        return(yhat)

# Load data in Pytorch 'Dataset' format
VitalArticles = WikiData(train, np.array(train_y))

# Split data
trainx, test = split(VitalArticles, val_prop = .05, seed = 344)

# Preprocess outcome label
train_y_ohe = np.zeros((len(train_y), len(input_data.keys())))
for idx,lbl in enumerate(train_y):
  train_y_ohe[idx, lbl] = 1
# Class weights
cw = torch.tensor(np.round(np.max(np.sum(train_y_ohe, axis=0)) / (np.sum(train_y_ohe, axis=0)), 1)).type(torch.float).to(device)

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
def convNN_search(parameters):
  """Set up, run and evaluate a baseline neural network"""
  # Get conv layers and preprocess
  cv = parameters.get("conv_layers")
  n_cv = cv.get("conv_layers")
  f_size = tuple([list(fs.values())[0] for fs in cv.get("filter_sizes")])
    # Split into train/test set
  train_current, test_current = split(trainx, val_prop = .05) 
  # CV with skorch
  net = NeuralNet(
    # Module
    module=Convolutions,
    # Module settings
    module__weights = FTEMB, # These are word embeddings
    module__num_classes = len(catmap),
    module__max_seq_len = args.seq_max_len,
    module__conv_filters = parameters["filters"],
    module__conv_layers = n_cv,
    module__filter_size = f_size,
    module__batch_norm = parameters["use_batch_norm"],
    module__p_dropout = parameters["dropout"],
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
    'filters': hp.choice('filters', [100, 150, 200, 250]),
    'conv_layers': hp.choice("conv_layers", 
                             [ {"conv_layers": 1,
                               "filter_sizes": [{"filter1": hp.uniformint("filter1-1",1, 15)}]},
                               {"conv_layers": 2,
                               "filter_sizes": [{"filter1": hp.uniformint("filter1-2",1, 15)},
                                                {"filter2": hp.uniformint("filter2-2",1, 15)}]},
                               {"conv_layers": 3,
                               "filter_sizes": [{"filter1": hp.uniformint("filter1-3",1, 15)},
                                                {"filter2": hp.uniformint("filter2-3",1, 15)},
                                                {"filter3": hp.uniformint("filter3-3",1, 15)}]}]),
    
    'optimizer': hp.choice("optimizer", ["Adam", "RMSprop"]),
    'dropout': hp.choice("dropout", [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.02)),
    'use_batch_norm': hp.choice('batch_norm', [True, False])
}

# Test if works
from hyperopt.pyll.stochastic import sample
parameters = sample(space)
po = convNN_search(parameters)

# Algorithm
tpe_algorithm = tpe.suggest

# Trials object to track progress
bayes_trials = Trials()
# File to save first results
with open(args.out_file, 'w') as of_connection:
  writer = csv.writer(of_connection)
  # Write the headers to the file
  writer.writerow(['loss', 'params', 'iteration', 'accuracy', "f1", "precision", "recall"])

# Optimize
best = fmin(fn = convNN_search, space = space, algo = tpe.suggest, 
            max_evals = args.max_evals, trials = bayes_trials)
            
# Run the model with the best paramaters
net = NeuralNet(
  # Module
  module=Convolutions,
  # Module settings
  module__weights = FTEMB, # These are word embeddings
  module__num_classes = len(catmap),
  module__max_seq_len = args.seq_max_len,
  module__conv_filters = 150,
  module__conv_layers = 3,
  module__filter_size = (1,1,4),
  module__batch_norm = False,
  module__p_dropout = 0.05,
  # Epochs & learning rate
  max_epochs=25,
  lr=0.00294,
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

# New text
WW = """Badr Hari[3] (Arabic: born 8 December 1984)[4] is a Moroccan-Dutch[5] super heavyweight kickboxer from Amsterdam, fighting out of Mike's Gym in Oostzaan. He is a former K-1 Heavyweight champion (2007–2008),[6] It's Showtime Heavyweight world champion (2009–2010)[7] and K-1 World Grand Prix 2008 and 2009 finalist.[8] Hari has been a prominent figure in the world of kickboxing; however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.[9][10] Hari has been officially praised by the King of Morocco, Mohammed VI, since 2009 for his outstanding accomplishments in the sport.[11] In April 2019, Hari was suspended for 19 months for a positive drug test."""
CL = ClassificationPipeline(tokenizer, net, args.seq_max_len)
proba, yclass = CL.predict([WW])
catmap[int(yclass)]

KB = """Kickboxing is a group of stand-up combat sports based on kicking and punching, historically developed from karate mixed with boxing.[2][3] Kickboxing is practiced for self-defence, general fitness, or as a contact sport.[4][5][6][7] Japanese kickboxing originated in the late 1950s, with competitions held since then.[8][9][10][11] American kickboxing originated in the 1970s and was brought to prominence in September 1974, when the Professional Karate Association (PKA) held the first World Championships. Historically, kickboxing can be considered a hybrid martial art formed from the combination of elements of various traditional styles. This approach became increasingly popular since the 1970s, and since the 1990s, kickboxing has contributed to the emergence of mixed martial arts via further hybridization with ground fighting techniques from Brazilian jiu-jitsu and folk wrestling."""
proba, yclass = CL.predict([KB])
catmap[int(yclass)]
