#%% Train the HAN
from HAN import HAN, Embedding_FastText, WikiDocData, batcher, process_batch
# Load pre-processed wikipedia data
import pickle
import numpy as np
import uuid
import os
import syntok.segmenter as segmenter
from tqdm import tqdm

# Import namespace
from argparse import Namespace

# Helper functions
from preprocess_utils import tokenize_text
import os

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

# Load data
inputs = {"History":{}, "Geography":{}, "Philosophy_and_religion":{}, 
          "Mathematics": {}, "Arts": {}, "Technology": {}, "Society_and_social_sciences":{}, 
          "Everyday_life": {}, "Physical_sciences": {}, "People": {},
          "Biology_and_health_sciences": {}}
failed = []
previous_docnr = "default"
# Max sentence length
MAX_SENT_LENGTH = 10
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

# Look
print(docs_vectorized[0])

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

#%% View the max length of all sentences in all documents

# Max length of the sentences
# (itertools.chain(*X)) makes list of lists into one, flat list
max_seq_len = max([len(seq) for seq in itertools.chain(*docs_vectorized)])

# Max length of documents (shoudl all be the same)
max_seq_doc = max([len(doc) for doc in docs_vectorized])

# View
print(len(docs_vectorized))
print(len(labels))

