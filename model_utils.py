# Utility functions for modeling go here

from keras.models import load_model
from keras.optimizers import Adam
from keras import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pickle
from preprocess_utils import tokenize_text
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn import metrics
    
# Load FastText embedding
def load_FT(path, word_index, embedding_dim, vocab_size, init_random_missing = False):
    """Load and process glove embeddings"""
    # Load word embeddings file
    embeddings_index = {}
    # Avg. embedding mean/var
    emb_mean = 0
    emb_var = 0
    with open(path, "r",  encoding='utf-8', newline='\n', errors='ignore') as inFile:
        for idx, line in enumerate(inFile):
            values = line.rstrip().split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            emb_mean += (np.mean(coefs) - emb_mean) / (idx + 1)
            emb_var += (np.var(coefs) - emb_var) / (idx + 1)
    # Turn into a matrix
    not_found = 0
    # Numpy matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i < vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                not_found += 1
                if init_random_missing:
                  # Random initialization of embedding
                  embedding_matrix[i] = np.random.normal(loc = emb_mean, scale = emb_var, size = embedding_dim)
    print("Could not find {} tokens in FastText ...".format(not_found))
    # Return
    return(embedding_matrix)

# Plot loss function
def plot_loss(history_dict, file_name = None):
    """Plot loss and save to directory"""
    loss_values = history_dict['train_loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if file_name is not None:
      plt.savefig(file_name)
    plt.show()

# Plot accuracy
def plot_accuracy(history_dict, file_name = None):
    """Plot accuracy and save to directory"""
    acc_values = history_dict['train_acc']
    val_acc_values = history_dict['val_acc']
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    if file_name is not None:
      plt.savefig(file_name)
    plt.show()

# Given a model & tokenizer, classify a bunch of texts
class ClassificationPipeline(object):
    def __init__(self, tokenizer, model, max_seq_len):
        """Pipeline used for classification of new texts"""
        self._tokenizer = tokenizer
        self._model = model
        self._max_seq_len = max_seq_len
    def predict(self, texts):
        """Input texts. Output labels."""
        # Keep track of records that failed to preprocess properly
        failed_preprocess = [True if text is None else False for text in texts]
        # Text to sequences
        seqs = self._tokenizer.texts_to_sequences([text for text in texts if text is not None])
        # Pad
        self._seqs_cap = preprocessing.sequence.pad_sequences(seqs, maxlen=self._max_seq_len)
        # Predict
        with torch.no_grad():
          self._model.eval()
          yhat_test = self._model(torch.tensor(self._seq_seq).type(torch.long))
        # Predict
        prob, yhat_class = yhat_test.max(axis=1) 
        # Return
        return(prob, yhat_class)
        
# Load tokenizer
def load_tokenizer(path):
    """Load Keras tokenizer"""
    with open(path, "rb") as inFile:
        tok = pickle.load(inFile)
    # Return
    return(tok)

# Create FastText embedding for PyTorch
def Embedding_FastText(weights, freeze_layer = True):
    """Set up a pytorch embedding matrix"""
    examples, embedding_dim = weights.shape
    # Set up layer
    embedding = nn.Embedding(examples, embedding_dim)
    # Add weights
    embedding.load_state_dict({"weight": torch.tensor(weights)})
    # If not trainable, set option
    if freeze_layer:
        embedding.weight.requires_grad = False
    # Return
    return(embedding)

# Dataset class
# This is used for the training loop (e.g. to create mini batches)
class WikiData(Dataset):
    def __init__(self, X, y):
        # Assert tensors
        assert type(X) == np.ndarray, "'X' must be numpy array"
        assert type(y) == np.ndarray, "'y' must be numpy array"
        # Must be same length
        assert X.shape[0] == y.shape[0], "'X' and 'y' different lengths"
        self.X = X
        self.y = y
        self.len = X.shape[0]
    def __getitem__(self, index):
        return(torch.tensor(self.X[index,:]).type(torch.long), torch.tensor(self.y[index]).type(torch.long))
    def __len__(self):
        return(self.len)
    
# Function that makes a sample for training and testing
def split(dataset, val_prop = .1, seed = None):
    """Split data into train / test"""
    X, y = dataset.X, dataset.y
    # Permute
    if seed is None:
        rp = np.random.permutation(y.shape[0])
    else:
        np.random.seed(seed)
        rp = np.random.permutation(y.shape[0])
    # Shuffle
    X = X[rp,:]
    y = y[rp]
    # Props to int
    n = y.shape[0]
    n_val = int(np.floor(val_prop * n))
    n_train = n - n_val
    # Subset into train, test
    X_train, y_train = X[0:n_train,:], y[0:n_train]
    X_val, y_val = X[n_train:(n_train + n_val),:], y[n_train:(n_train + n_val)]
    # Create new dataset instantiation for train and val
    return(WikiData(X_train, y_train), WikiData(X_val, y_val))
