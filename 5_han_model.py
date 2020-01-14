#%% Train the HAN
from HAN import HAN, Embedding_FastText, WikiDocData, batcher, process_batch, split_data
# Load pre-processed wikipedia data
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

if not os.path.exists("data/HAN_wiki_preprocessed.pickle"):
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
    # Save
    with open("data/HAN_wiki_preprocessed.pickle", "wb") as outFile:
        pickle.dump(inputs, outFile, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open("data/HAN_wiki_preprocessed.pickle", "rb") as inFile:
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

if not os.path.exists("embeddings/HAN_embeddings.pickle"):
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
    with open("embeddings/HAN_embeddings.pickle", "wb") as outFile:
        pickle.dump(FTEMB, outFile, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open("embeddings/HAN_embeddings.pickle", "rb") as inFile:
        FTEMB = pickle.load(inFile)

#%% View the max length of all sentences in all documents

# Max length of the sentences
# (itertools.chain(*X)) makes list of lists into one, flat list
max_seq_len = max([len(seq) for seq in itertools.chain(*docs_vectorized)])

# Max length of documents (shoudl all be the same)
max_seq_doc = max([len(doc) for doc in docs_vectorized])

# View
print((max_seq_len))
print((max_seq_doc))

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
WikiHAN = HAN(FTEMB, hidden_size, hidden_size, batch_size, num_classes)

# Set up optimizer
optimizer = optim.Adam(WikiHAN.parameters(), lr = 0.005)

# Criterion
criterion = nn.CrossEntropyLoss(weight=cw)

#%% Prepare train /test data

# Create batched data
train, val = split_data(docs_vectorized, labels_vect, 6754, p=0.05)
# Make dataset
test = WikiDocData(val[0], val[1])

#%% Train HAN

WikiHAN_out, history = train_han(train[0], train[1], WikiHAN, optimizer, criterion,
                                epochs = 2, val_split = 0.1, batch_size = batch_size,
                                device = device)

# %% Train model

def train_han(X, y, model, optimizer, criterion, epochs = 10, 
              val_split = .1, batch_size=64, device = "cpu"):
    """
    Train a Hierarchical Attention Network

    :param X: input documents. Structured as a list of lists, where one entry is a list of input sentences.
                all input sentences must be of the same size.
    :param y: numpy array containing the output labels
    :param model: a HAN model.
    :param optimizer: optimizer used for gradient descent.
    :param criterion: optimization criterion
    :param epochs: number of epochs to train the model.
    :param val_split: proportion of data points of total documents used for validation.
    :param batch_size: size of the minibatches.
    :param device: either one of 'cpu' or 'cuda' if GPU is available.

    :return: Tuple containing:
        1. Trained pytorch model
        2. Training history. Dict containing 'training_loss', 'training_acc' and 'validation_acc'
    """
    # Number of input examples
    n_examples = len(X)
    # Keep track of training loss / accuracy
    training_loss = []
    training_acc = []
    validation_acc = []
    # For each epoch, train the mopdel
    for epoch in range(0, epochs):
        running_loss = 0.0
        running_acc = 0.0
        # Split data
        batch_train, batch_val = split_data(train[0], train[1], p = val_split)
        # Make datasets
        batch_train_data = WikiDocData(batch_train[0], batch_train[1])
        batch_val_data = WikiDocData(batch_val[0], batch_val[1])
        # For each train/test example
        for i in range(n_examples // batch_size):
            model.train()
            # Draw a batch
            current_batch = batcher(batch_train_data, batch_size)
            # Process input batches
            #  What happens here is as follows:
            #   (1) all first sentences go with first sentences for all docs etc.
            #   (2) Apply packed_sequences to make variable-batch lengths
            seqs, lens = process_batch(current_batch, device = device)
            # GT labels
            labels_ground_truth = torch.tensor([b[1] for b in current_batch])
            # Zero gradients
            model.zero_grad()
            # Predict output
            predict_out = model(seqs, lens)
            # Get max
            predict_class = torch.argmax(predict_out, dim=1).cpu().numpy()
            # Loss
            loss_out = criterion(predict_out, labels_ground_truth)
            # As item
            loss_value = loss_out.item()
            # GT labels to numpy
            labels_ground_truth = labels_ground_truth.numpy()
            acc_batch = sum(predict_class == labels_ground_truth) / labels_ground_truth.shape[0]
            # Update loss and accuracy
            running_loss += (loss_value - running_loss) / (i + 1)
            running_acc += (acc_batch - running_acc) / (i + 1)
            # Print if desired
            if i % 5 == 0:
                print("Loss is {} on iteration {} for epoch {} ...".format(np.round(running_loss, 3), i, epoch))
            # Produce gradients
            loss_out.backward()
            # Make step
            optimizer.step()
        # Append loss
        training_loss.append(running_loss)
        training_acc.append(running_acc)
        # On validation data
        with torch.no_grad():
            model.eval()
            io = batcher(batch_val_data, len(batch_val_data.X))
            # Process batches
            seqs, lens = process_batch(io, device = device)
            out = torch.argmax(model(seqs, lens), dim=1)
        # Process true label
        ytrue = [doc[1] for doc in io]
        ytrue = torch.tensor(ytrue).numpy()
        # Acc
        val_acc = np.round(sum(out.numpy() == ytrue) / ytrue.shape[0], 3)
        validation_acc.append(val_acc)
        # Print
        print("-------------")
        print("Training Loss is {} at epoch {} ...".format(np.round(running_loss, 3), epoch))
        print("Training accuracy is {} at epoch {} ...".format(np.round(running_acc, 3), epoch))
        print("Validation accuracy is {} at epoch {} ...".format(val_acc, epoch))
        print("-------------")

    # Return
    return(model, {"training_loss": training_loss, "training_acc": training_acc, "validation_acc": validation_acc})

# %%
