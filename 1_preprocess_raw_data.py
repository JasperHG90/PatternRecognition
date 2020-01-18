# Preprocess input data

# Load pre-processed wikipedia data
import pickle
import numpy as np
import syntok.segmenter as segmenter
from keras.preprocessing.text import Tokenizer # Use keras for tokenization & preprocessing
from keras import preprocessing
import matplotlib.pyplot as plt
from model_utils import load_FT, WikiData, split
from tqdm import tqdm
from argparse import Namespace
from preprocess_utils import tokenize_text
import os
import itertools

## MAKE SURE TO SHUFFLE DATA

# Settings
args = Namespace(
    # Directories
    data_dir_other="data/other",
    data_dir_han="data/HAN",
    # Raw data
    data_raw="data/raw/WikiEssentials_L4.txt",
    # Name of the output data
    data_out_name="WikiEssentials_L4_processed",
    # Tokenization settings
    token_lower=False,
    token_remove_digits=True,
    # Number of sentences to process for each document
    max_sent_length_other=8,
    max_sent_length_han=[8, 10, 12, 15],
    # Keras tokenizer settings
    input_vocabulary_size=20000, # Number of tokens to keep
    # Random seed for data split
    seed=67976,
    # Test proportion size
    test_proportion=0.1,
    # Embedding dim
    embedding_dim = 300
)

#%% Read data and preprocess for baseline / CNN

# To store results
inputs = {"History":{}, "Geography":{}, "Philosophy_and_religion":{}, 
          "Mathematics": {}, "Arts": {}, "Technology": {}, "Society_and_social_sciences":{}, 
          "Everyday_life": {}, "Physical_sciences": {}, "People": {},
          "Biology_and_health_sciences": {}}
# Store failed documents
failed = []
# Placeholder
previous_docnr = "default"
#stop = 100 # (for testing purposes)
with open(os.path.join(args.data_raw), "r", encoding="utf8") as inFile:
    # Counter for the number of sentences processed
    sentcount = 0
    # Capture sentences
    sent_level = list()
    # Identifier when doc is finished
    docfinish = False
    # Counter
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
        # Else, use syntok to segment a paragraph into sentences
        a = segmenter.process(lnsp[-1])
        # For each paragraph, do ...
        for par in a:
            # For each sentence in the paragraph, do ...
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
                    sent_level.append(csent)
                    sentcount += 1
                    if sentcount > args.max_sent_length_other:
                        # Take doc number
                        docnr = lnsp[0]
                        # Add to inputs
                        inputs[lbl0][docnr] = sent_level
                        # Set doc to finished
                        docfinish = True
                        break
        i += 1
# Inspect failed docs
print(len(failed))
# View one
index = 200
failed[index]

# How many items?
for k, v in inputs.items():
    print(k)
    print(len(v))
print(sum([len(v) for v in inputs.values()]))

# Save preprocessed data to disk
with open(os.path.join(args.data_dir_other,
                       '{}_P3_preprocessed.pickle'.format(args.data_out_name)),
          'wb') as handle:
    pickle.dump(inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Further preprocess data

train_x = []
train_y = []
catmap = {}

# For each
for idx, itms in enumerate(inputs.items()):
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

#%% Tokenize & vectorize data

# Create tokenizer
tokenizer = Tokenizer(num_words=args.input_vocabulary_size,
                      lower=False,
                      filters='!"$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')
# Fit on the documents
tokenizer.fit_on_texts(train_x)
# Vectorize text to sequences
x_train = tokenizer.texts_to_sequences(train_x)
# Average length of each sequence
seq_len = [len(x) for x in x_train]
print(np.median(seq_len))
print(np.max(seq_len))
# Store median length
args.seq_median_length = int(np.median(seq_len))
# Pad sequences
train = preprocessing.sequence.pad_sequences(x_train, maxlen=args.seq_median_length)

# Save tokenizer
with open(os.path.join(args.data_dir_other, "tokenizer.pickle"), 'wb') as handle:
  pickle.dump(tokenizer,  handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Create embedding

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
with open(os.path.join(args.data_dir_other, "embedding_matrix.pickle"), 'wb') as handle:
  pickle.dump(FTEMB,  handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Split into train/test data

np.random.seed(args.seed)
# Permutation
rp = np.random.permutation(len(train))

#%%

# Shuffle test / train data
train = train[rp, :]
train_y = np.array(train_y)
train_y = train_y[rp]

# Test proportion to number of examples of total
test_idx = int(np.floor(args.test_proportion * len(train)))
# Subset train and test
test_x = train[:test_idx, :]
train_x = train[test_idx:len(train), :]
test_y = train_y[:test_idx]
train_y = train_y[test_idx:len(train)]

# Assertions
assert test_x.shape[0] + train_x.shape[0] == 10016

#%% Save datasets

with open(os.path.join(args.data_dir_other, "vectorized_input_data.pickle"), 'wb') as handle:
  pickle.dump({"train_x":train_x, "test_x":test_x,
               "train_y":train_y, "test_y":test_y,
               "catmap":catmap},
              handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Data preprocessing for LSTM / HAN

for max_sent_len in args.max_sent_length_han:
    # Load data
    inputs = {"History":{}, "Geography":{}, "Philosophy_and_religion":{},
            "Mathematics": {}, "Arts": {}, "Technology": {}, "Society_and_social_sciences":{},
            "Everyday_life": {}, "Physical_sciences": {}, "People": {},
            "Biology_and_health_sciences": {}}
    failed = []
    previous_docnr = "default"
    #stop = 100
    with open(os.path.join(args.data_raw), "r", encoding="utf8") as inFile:
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
                if sentcount < 20:
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
                        if sentcount > max_sent_len:
                            # Take doc number
                            docnr = lnsp[0]
                            # Add to inputs
                            inputs[lbl0][docnr] = sent_level
                            # Set doc to finished
                            docfinish = True
                            break
            i += 1
    # Save
    with open(os.path.join(args.data_dir_han,
                           "HAN_wiki_preprocessed_S{}.pickle").format(max_sent_len),
              "wb") as outFile:
        pickle.dump(inputs, outFile, protocol=pickle.HIGHEST_PROTOCOL)

    # Reshape data susch that it is a nested list of:
    # --> documents
    #  --> sentences
    docs = []
    labels = []
    for label, documents in inputs.items():
        for doc_id, content in documents.items():
            docs.append(content)
            labels.append(label)

    # Flatten the inputs data
    inputs_flat = [txt for txt in itertools.chain(*docs)]

    # Create tokenizer
    tokenizer = Tokenizer(num_words=args.input_vocabulary_size,
                          lower=False,
                          filters='!"$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')

    # Fit on the documents
    tokenizer.fit_on_texts(inputs_flat)

    # Number of unique words
    word_index = tokenizer.word_index

    # Vectorize the documents (original 'docs' list)
    docs_vectorized = [tokenizer.texts_to_sequences(doc) for doc in docs]

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

    # Train/test
    docs_vectorized = [docs_vectorized[idx] for idx in rp]
    labels = [labels[idx] for idx in rp]
    # Split
    train_x = [docs_vectorized[idx] for idx in range(test_idx)]
    test_x = [docs_vectorized[idx] for idx in range(test_idx, len(docs_vectorized))]
    train_y = [labels[idx] for idx in range(test_idx)]
    test_y = [labels[idx] for idx in range(test_idx, len(docs_vectorized))]

    ### Load the embeddings
    # Get tokens to be looked up in FT embedding
    WI = {k:v for k,v in tokenizer.word_index.items() if v <= (args.input_vocabulary_size - 1)}
    FTEMB = load_FT("embeddings/wiki-news-300d-1M.vec", WI, 300, args.input_vocabulary_size)
    # Check which are 0
    io = np.sum(FTEMB, axis=1)
    zerovar = np.where(io == 0)[0]
    # Get words
    zerovar_words = {k:v for k,v in WI.items() if v in zerovar}
    # Save embedding
    with open(os.path.join(args.data_dir_han, "HAN_embeddings_S{}.pickle").format(max_sent_len), "wb") as outFile:
        pickle.dump(FTEMB, outFile, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.data_dir_han, "tokenizer_S{}.pickle").format(max_sent_len), "wb") as outFile:
        pickle.dump(tokenizer, outFile, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.data_dir_han, "tokenizer_S{}.pickle").format(max_sent_len), "wb") as outFile:
        pickle.dump(tokenizer, outFile, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.data_dir_han, "data_S{}.pickle").format(max_sent_len), "wb") as outFile:
        pickle.dump({"train_x": train_x,
                     "test_x": test_x,
                     "train_y": train_y,
                     "test_y": test_y,
                     "labels_vectorized": labels_vect,
                     "labels_to_idx": label_to_idx,
                     "idx_to_label": idx_to_label},
                    outFile, protocol=pickle.HIGHEST_PROTOCOL)
