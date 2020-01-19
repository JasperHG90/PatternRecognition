#%% This script preprocesses the wikipedia data and trains the LSTM Network.
# The LSTM Network utility functions can be found in LSTM.py

from LSTM import LSTMN, WikiDocData, split_data, train_lstmn, batcher, process_batch
import pickle
import numpy as np
import uuid
import os
import syntok.segmenter as segmenter
from tqdm import tqdm
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import optim
import torch.nn as nn
from sklearn import metrics

# Import namespace
from argparse import Namespace

# Helper functions
from preprocess_utils import tokenize_text
import os
import pickle

# Set up namespace
args = Namespace(
    # Input data settings
    data_dir = "data",
    data_prefix = "WikiEssentials_L4",
    # Preprocessing settings
    token_lower = False,
    token_remove_digits = True,
    # Tokenizer etc.
    max_vocab_size = 20000
)

# Run preprocess function
path_to_file = os.path.join(args.data_dir, args.data_prefix + ".txt")

#%% Preprocess the input data

if not os.path.exists("data/LSTMN_wiki_preprocessed.pickle"):
    # Load data
    inputs = {"History":{}, "Geography":{}, "Philosophy_and_religion":{}, 
            "Mathematics": {}, "Arts": {}, "Technology": {}, "Society_and_social_sciences":{}, 
            "Everyday_life": {}, "Physical_sciences": {}, "People": {},
            "Biology_and_health_sciences": {}}
    failed = []
    previous_docnr = "default"
    # Max sentence length
    MAX_SENT_LENGTH = 8
    #stop = 100
    with open(path_to_file, "r", encoding="utf8") as inFile:
        # Counter for the number of sentences processed
        sentcount = 0
        # Capture sentences
        sent_level = list()
        # Identifier when doc is finished
        docfinish = False
        i = 0
        # Read lines
        for line in tqdm(inFile):
            #if i == stop:
            #    break
            if i == 0:
                i += 1
                continue
            # Split at first whitespace
            lnsp = line.split("\t")
            # Take label
            lbl = lnsp[1]
            # Split labels
            lblsp = lbl.split("+")
            lbl0 = lblsp[0]
            # Get doc number
            if lnsp[0] != previous_docnr:
                if sentcount < 15:
                    # Take doc number
                    docnr = lnsp[0]
                    # Add to inputs
                    inputs[lbl0][docnr] = sent_level                
                    # Reset
                    sent_level = list()
                    sentcount = 0
                    docfinish = False
            # Set previous doc number to current
            previous_docnr = lnsp[0]
            # Process each sentence of paragraph, unless already have enough sentences
            if docfinish: continue
            a = segmenter.process(lnsp[-1])
            for par in a:
                for sent in par:
                    csent = "".join([token.spacing + token.value for token in sent]).strip()
                    # Tokenize text
                    txt_tok = tokenize_text(csent,
                                            lower_tokens=args.token_lower,
                                            remove_digits_token=args.token_remove_digits)
                    # If none, pass ...
                    if txt_tok is None:
                        failed.append(csent)
                        continue
                    else:
                        sent_level.append(txt_tok)
                        sentcount += 1
                        if sentcount > MAX_SENT_LENGTH:
                            # Take doc number
                            docnr = lnsp[0]
                            # Add to inputs
                            inputs[lbl0][docnr] = sent_level
                            # Set doc to finished
                            docfinish = True
                            break
            i += 1
    # Save
    with open("data/LSTMN_wiki_preprocessed.pickle", "wb") as outFile:
        pickle.dump(inputs, outFile, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open("data/LSTMN_wiki_preprocessed.pickle", "rb") as inFile:
        inputs = pickle.load(inFile)

# How many items?
for k, v in inputs.items():
    print(k)
    print(len(v))
print("-------")
print("Total items: " + str(sum([len(v) for v in inputs.values()])))

#%% Further preprocessing

# Reshape data susch that it is a nested list of:
# --> documents
#  --> sentences
docs = []
labels = []
for label, documents in inputs.items():
    for doc_id, content in documents.items():
        docs.append(content)
        labels.append(label)

# View
docs[0]

#%% Set up the tokenizer

# First, create a tokenizer and choose 20.000 most common words
from keras.preprocessing.text import Tokenizer # Use keras for tokenization & preprocessing
import itertools

# Flatten the inputs data
inputs_flat = [txt for txt in itertools.chain(*docs)]

# Create tokenizer
tokenizer = Tokenizer(num_words=args.max_vocab_size,
                      lower = False,
                      filters = '!"$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')

# Fit on the documents
tokenizer.fit_on_texts(inputs_flat)

# Number of unique words
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#%% Vectorize the documents

# Vectorize the documents (original 'docs' list)
docs_vectorized = [tokenizer.texts_to_sequences(doc) for doc in docs]
docs_vectorized_lstm = []
doc_lstm = []
for doc in docs_vectorized:
    doc_lstm = []
    for sent in doc:
        doc_lstm = doc_lstm + sent
    docs_vectorized_lstm.append(doc_lstm)
    

# Look
print(docs_vectorized[0])
print(docs_vectorized_lstm[0])

# Vectorize outcome labels
label_to_idx = {}
idx_to_label = {}
labels_vect = []
i = 0
for label in labels:
    if label_to_idx.get(label) is None:
        label_to_idx[label] = i
        idx_to_label[i] = label
        i += 1
    labels_vect.append(label_to_idx[label])

# View
label_to_idx

#%% Set up the embedding

if not os.path.exists("embeddings/LSTMN_embeddings.pickle"):
    ### Load the embeddings
    from model_utils import load_FT
    # Get tokens to be looked up in FT embedding
    WI = {k:v for k,v in tokenizer.word_index.items() if v <= (args.max_vocab_size - 1)}
    FTEMB = load_FT("embeddings/wiki-news-300d-1M.vec", WI, 300, args.max_vocab_size)
    # Check which are 0
    io = np.sum(FTEMB, axis=1)
    zerovar = np.where(io == 0)[0]
    # Get words
    zerovar_words = {k:v for k,v in WI.items() if v in zerovar}
    zerovar_words
    # Save mbedding
    with open("embeddings/LSTMN_embeddings.pickle", "wb") as outFile:
        pickle.dump(FTEMB, outFile, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open("embeddings/LSTMN_embeddings.pickle", "rb") as inFile:
        FTEMB = pickle.load(inFile)

#%% View the max length of all sentences in all documents

# Max length of the sentences
# (itertools.chain(*X)) makes list of lists into one, flat list
max_seq_len = max([len(seq) for seq in itertools.chain(*docs_vectorized)])
max_seq_len_lstm = max([len(seq) for seq in docs_vectorized_lstm])

# Max length of documents (shoudl all be the same)
max_seq_doc = max([len(doc) for doc in docs_vectorized])
# No point to replicate here for LSTM

# View
print((max_seq_len))
print((max_seq_doc))
print((max_seq_len_lstm))


#%% Class weights

cw = [len(v) for k,v in inputs.items()]
cw = np.max(cw) / cw
cw = torch.tensor(cw).type(torch.float).to(device)

#%% Set up the model

# Hidden size
hidden_size = 64
batch_size = 128
num_classes = len(np.unique(labels_vect))

# Set up the model
WikiLSTMN = LSTMN(FTEMB, batch_size, num_classes)

# Set up optimizer
optimizer = optim.Adam(WikiLSTMN.parameters(), lr = 0.004)

# Criterion
criterion = nn.CrossEntropyLoss(weight=cw)

#%% Prepare train /test data

# Create batched data
train, val = split_data(docs_vectorized, labels_vect, 6754, p=0.05)
train_lstm, val_lstm = split_data(docs_vectorized_lstm, labels_vect, 6754, p=0.05)
# Make dataset
test = WikiDocData(val[0], val[1])
test_lstm = WikiDocData(val_lstm[0], val_lstm[1])

#%% Train LSTMN

WikiLSTMN = LSTMN(FTEMB, batch_size, num_classes)
WikiLSTMN = WikiLSTMN.to(device)

WikiLSTMN_out, history = train_lstmn(train_lstm[0], train_lstm[1], WikiLSTMN, optimizer, criterion,
                                epochs = 5, val_split = 0.1, batch_size = batch_size,
                                device = device)

#%% Evaluate the model on test data

# For now, just make a single batch of the test data for evaluation
valbatch = batcher(test, len(test.X))

# Preprocess
seqs, lens = process_batch(valbatch, device = device)

# Predict
with torch.no_grad():
    WikiLSTMN_out.eval()
    probs = WikiLSTMN_out(seqs, lens)

# %% Classes

# To classes
out = torch.argmax(probs, dim=1).numpy()

# Get true label
ytrue = [batch[1] for batch in valbatch]
ytrue = torch.tensor(ytrue).numpy()

# Accuracy
sum(out == ytrue)/len(out)

#%% Print classification report

# Print classification report
print(metrics.classification_report(ytrue, out, target_names = list(label_to_idx.keys())))

# %% Get attention weights

# Predict
with torch.no_grad():
    WikiLSTMN_out.eval()
    probs, attn = WikiLSTMN_out(seqs, lens, return_attention_weights = True)

# %% Preprocess attention weights

word_weights, sent_weights = attn
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
idx_to_word = {v:k for k,v in word_index.items()}

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
sa = sentence_attention(sent_weights[7][doc_idx,:,:].numpy())
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
