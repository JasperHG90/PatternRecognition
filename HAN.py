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
from model_utils import load_FT, Embedding_FastText, WikiData, split
from collections import defaultdict

#%%

# Details for this script
from argparse import Namespace

# Model settings
args = Namespace(
  # File to save results
  out_file = 'results/HAN_trials.csv',
  # Number of times to evaluate bayesian search for hyperparams
  max_evals = 200,
  # Number of input sentences
  number_of_sentences = 8,
  # Size of the vocabulary
  input_vocabulary_size = 15000,
  # Embedding size
  embedding_dim = 300,
  # Max length of text sequences
  seq_max_len = 20,
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
doc_id_map = []
catmap = {}

doc_id_lookup = 0
# For each
for idx, itms in enumerate(input_data.items()):
  # Label and texts
  cat = idx
  txts = itms[1]
  catmap[cat] = itms[0]
  # For each text, append
  for doc_id, txt_lst in txts.items():
    xo= 0
    #if len(txt_lst) < 3:
    #  continue
    par_out = []
    for idx_sentence, txt in enumerate(txt_lst):
      if xo == 8:
        xo = 0
        break
      train_x.append(txt)
      xo += 1
      doc_id_map.append(doc_id_lookup)
    train_x.append(" ".join(par_out).replace("'s", ""))
    train_y.append(cat)
    doc_id_lookup += 1

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

# Number of unique words
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# To sequence (vectorize)
x_train = tokenizer.texts_to_sequences(train_x)

# Average length
seq_len = [len(x) for x in x_train]
print(np.median(seq_len))
print(np.max(seq_len))

# Pad sequences
train = preprocessing.sequence.pad_sequences(x_train, maxlen=args.seq_max_len)

#%% HAN embeddings

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
with open("embeddings/HAN_prep.pickle", 'wb') as handle:
  pickle.dump(FTEMB,  handle, protocol=pickle.HIGHEST_PROTOCOL)

# Get tokens to be looked up in FT embedding
with open("embeddings/HAN_prep.pickle", "rb") as inFile:
    FTEMB = pickle.load(inFile)

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
        sentence = torch.sum(torch.mul(alphas, hidden_states), dim=1)
        #for idx in range(0, hidden_states.size(0)):
            # Get hidden state at time t
        #    hidden_current = hidden_states[idx]
            # Get attention weights at time t
        #    alphas_current = alphas[idx]
            # Hadamard product (element-wise)
        #    vector_weighted = hidden_current * alphas_current
            # Concatenate
        #    if idx > 0:
        #        s = torch.cat((s, vector_weighted), 0)
        #    else:
        #        s = vector_weighted
        # Sum across time axis (0)
        #return(torch.sum(s, 0))
        # Return
        return(sentence)

#%% Word encoder

class word_encoder(nn.Module):
    def __init__(self, weights, hidden_size):
        super(word_encoder, self).__init__()
        self._hidden_size = hidden_size
        self._weight_input_size = weights.shape[1]
        # Embedding
        self.embedding = Embedding_FastText(weights, freeze_layer = True)
        # Word-GRU
        self.GRU = nn.GRU(self._weight_input_size, self._hidden_size,
                        bidirectional=True, batch_first=True)       
        # Attention
        self.attention = Attention(self._hidden_size)
    def forward(self, inputs):
        """
        :param inputs: input document with dim (sentence x seq_length)
        """
        # Embedding
        embedded = self.embedding(inputs)
        # Bidirectional GRU
        output_gru, _ = self.GRU(embedded)
        # Attention
        output_attention = self.attention(output_gru)
        # Return
        return(output_attention.unsqueeze(dim=1))

class sentence_encoder(nn.Module):
    def __init__(self, hidden_size):
        super(sentence_encoder, self).__init__()
        self._hidden_size = hidden_size
        # Sentence-GRU
        self.GRU = nn.GRU(self._hidden_size * 2, self._hidden_size,
                        bidirectional=True, batch_first=True)       
        # Attention
        self.attention = Attention(hidden_size)
    def forward(self, encoder_output):
        """
        :param encoder_output: output of the word encoder with shape (sentences x 1 x 2 * hidden_dim)
        """
        # Bidirectional GRU
        output_gru, _ = self.GRU(encoder_output)
        # Attention
        #  Permute output_gru s.t. we go from (sentences, 1, 2*hidden_dim) to
        #  (1, sentences, 2*hidden_dim). This ensures that we take sum across
        #  sentences.
        output_attention = self.attention(output_gru.permute(1, 0, 2))
        # Return
        return(output_attention) 

class HAN(nn.Module):
    def __init__(self, weights, hidden_size, num_classes):
        super(HAN, self).__init__()
        self._hidden_size = hidden_size
        self._embedding_dim = weights.shape
        self._num_classes = num_classes
        # Set up word encoder
        self._word_encoder = word_encoder(weights, self._hidden_size)
        # Set up sentence encoder
        self._sentence_encoder = sentence_encoder(self._hidden_size)
        # Set up a linear layer
        self._linear1 = nn.Linear(self._hidden_size * 2, self._num_classes)
    def forward(self, input_document):
        """
        :param input_document: input document with dim (sentence x seq_length)
        """
        # Put document through word encoder
        sentences_encoded = self._word_encoder(input_document)
        # Put sentences through sentence encoder
        document_encoded = self._sentence_encoder(sentences_encoded)
        # Linear layer & softmax
        prediction_out = F.softmax(self._linear1(document_encoded.flatten()))
        # Return
        return(prediction_out)

#%% WikiData class for HAN

class WikiData(Dataset):
    def __init__(self, document_ids, X, y):
        # Assert tensors
        assert type(X) == np.ndarray, "'X' must be numpy array"
        assert type(y) == np.ndarray, "'y' must be numpy array"
        # Reshape input based on doc ids
        X_3D = np.zeros((len(np.unique(document_ids)), args.number_of_sentences, args.seq_max_len), dtype = np.int)
        # Insert each document into the new input array
        document_idx_prev = None
        sentence_idx = 0
        for total_idx, document_idx in enumerate(document_ids):
            if document_idx == document_idx_prev:
                sentence_idx += 1
            else:
                sentence_idx = 0
            # Insert
            X_3D[document_idx, sentence_idx, :] = X[total_idx,:]
            document_idx_prev = document_idx
        # Must be same length
        assert X_3D.shape[0] == y.shape[0], "'X' and 'y' different lengths"
        self.X = X_3D
        self.y = y
        self.len = X_3D.shape[0]
    def __getitem__(self, index):
        return(torch.tensor(self.X[index,:,:]).type(torch.long), torch.tensor(self.y[index]).type(torch.long))
    def __len__(self):
        return(self.len)

#%%

# Load data in Pytorch 'Dataset' format
# See 'model_utils.py'
VitalArticles = WikiData(doc_id_map ,train, np.array(train_y))

#%%

# Class weights
# These weights are unnormalized but that's what pytorch is expecting
cw = torch.tensor(np.max(np.sum(train_y_ohe, axis=0)) / (np.sum(train_y_ohe, axis=0))).type(torch.float).to(device)

#%% Reshape input document

idx = [i for i in range(2320)]
Xy = [VitalArticles.__getitem__(i) for i in idx]
# Random permutation
np.random.seed(6782)
perm = np.random.permutation(len(Xy))
Xy = [Xy[idx] for idx in perm]

#%% Make weights

cw = torch.tensor(np.max(np.sum(train_y_ohe[:,0:3], axis=0)) / (np.sum(train_y_ohe[:,0:3], axis=0))).type(torch.float).to(device)

#%% Train function

# Set up the model
WikiHAN = HAN(FTEMB, 64, 3)

# Set up optimizer
optimizer = optim.Adam(WikiHAN.parameters(), lr = 0.0001)

# Criterion
criterion = nn.CrossEntropyLoss()

# Track loss
train_loss = []

#%%

# Epochs
for epoch in range(0, 3):
    running_loss = 0.0
    # For each train/test example
    for idx, input in enumerate(Xy):

        WikiHAN.train()

        # Input document
        input_doc = input[0]
        input_class = input[1]

        # Zero gradients
        WikiHAN.zero_grad()

        # Predict output
        predict_out = WikiHAN(input_doc)

        # Loss
        loss_out = criterion(predict_out.unsqueeze(0), input_class.unsqueeze(0))
        # As item
        loss_value = loss_out.item()
        running_loss += (loss_value - running_loss) / (idx + 1)
        if idx % 100 == 0:
            print("Loss is {} on iteration {} for epoch {} ...".format(running_loss, idx, epoch))

        # Produce gradients
        loss_out.backward()

        # Make step
        optimizer.step()

    # Append loss
    train_loss.append(running_loss)

    # Predict
    pred = []
    with torch.no_grad():
        WikiHAN.eval()
        for idx, input in enumerate(Xy):
            out = WikiHAN(input[0])
            pred.append(torch.argmax(out).numpy())

    pred = np.vstack(pred).squeeze()
    ytrue = VitalArticles.y[:2320]

    # Print
    print("-------------")
    print("Loss is {} at epoch {} ...".format(running_loss, epoch))
    print("Accuracy is {} at epoch {} ...".format(np.sum(pred == ytrue) / ytrue.shape[0],epoch))
    print("-------------")

#%% Run the training function

pred = []
with torch.no_grad():
    WikiHAN.eval()
    for idx, input in enumerate(Xy):
        out = WikiHAN(input[0])
        pred.append(torch.argmax(out).numpy())

#%% To NP

pred = np.vstack(pred).squeeze()
ytrue = VitalArticles.y[:2320]

#%% Set up a HAN



#%% Run doc through HAN

HAN_out = WikiHAN(inputs_x)

#%% Word encoder

WE = word_encoder(FTEMB, 32)
sentences = WE(inputs_x)

#%% Sentence encoder

SE = sentence_encoder(32)
document = SE(sentences)

#%%

# Set up an embedding
embedding = Embedding_FastText(FTEMB, freeze_layer = True)    
Encoder_GRU = nn.GRU(FTEMB.shape[1], 32, bidirectional = True, batch_first= True)



#%%

#inputs_stacked = torch.stack(inputs_x, 0)
#inputs_stacked.shape

#%%

# Embed
emb_out = embedding(inputs_x)
emb_out.shape

#%%

# GRU
GRU_out, hidden_out = Encoder_GRU(emb_out)
# NB: GRU_out contains the hidden states for 1,...,T
#      (https://discuss.pytorch.org/t/how-to-retrieve-hidden-states-for-all-time-steps-in-lstm-or-bilstm/1087/14)
# DIM: (directions, input_sentence_length, hidden_dim)
GRU_out.shape

#%% Attention layer

att = Attention(32)

#%% Across GRU 

GRU_attended = att(GRU_out)
GRU_attended.shape

#%% Sentence-level GRU

Sentence_GRU = nn.GRU(64, 32, bidirectional=True)

#%% Run over GRU_attended

GRU_sentence, _ = Sentence_GRU(GRU_attended.unsqueeze(dim=0))

#%% Sentence attention

att_sentence = Attention(32)

#%% Sentence attention

GRU_sentence_attended = att_sentence(GRU_sentence)

#%%

layer1 = nn.Linear(2 * 32, 2 * 32)

#%%

layer2 = nn.Linear(2 * 32, 2 * 32, bias = False)

#%%

GRU_attended = layer1(GRU_out)

#%%

GRU_attended = F.tanh(GRU_attended)

#%%

attention_weights = F.softmax(layer2(GRU_attended), dim=1)
attention_weights.shape

#%% Torch mul

sentence = torch.mul(GRU_out, attention_weights)
sentence = torch.sum(sentence, dim=1)
# DIM: (batch_size, 2*hidden_dim)
sentence.shape

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
