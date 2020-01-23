## Implementation of a Long-Short Term Memory Network (LSTM Network)
##
##This implementation is based on the HAN implementation by Jasper Ginn <j.h.ginn@uu.nl> for the same project
##
## Written by: Luis Martín-Roldán Cervantes <l.martin-roldancervantes@students.uu.nl>
## Course: Pattern Recognition @ Utrecht University

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from model_utils import Embedding_FastText
import numpy as np
from sklearn import metrics
# This is a technical thing
# See stackoverflow:
#   - PyTorch: training with GPU gives worse error than training the same thing with CPU
torch.backends.cudnn.enabled = False

padding_symbol = '<PAD>'
"""
LSTM utility functions:
    To make the LSTM Network self-contained, I put all utility functions in this python file. The preprocessing
    steps and dataset construction are a little different from the other models. The preprocessing 
    functions are as follows:
        1. Embedding_FastText: creates a Pytorch embedding layer from pre-trained weights
		2. WikiDocData: Pytorch Dataset used to store & retrieve wikipedia data.
        3. batcher: function that creates minibatches for training
        4. process_batch: processes a minibatch of wikipedia articles.
        5. split_data: function that splits wikipedia data into train & test
        6. train_lstmn: training regime for the LSTM Network.
"""

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

# Create a dataset to hold both the documents and the labels
class WikiDocData(Dataset):
    def __init__(self, X, y):
        # Must be same length
        assert len(X) == len(y), "'X' and 'y' different lengths"
        self.X = X
        self.y = y
        self.len = len(X)
    def __getitem__(self, index):
        # Retrieve X
        X = torch.tensor([self.X[index]]).type(torch.long)
        # Each sentence to tensor
        return((X, 
                torch.tensor(self.y[index]).type(torch.long)))
    def __len__(self):
        return(self.len)

# Create function that makes a minibatch
def batcher(wiki_data, batch_size):
    """
    Create a minibatch from WikiDocData dataset
    """
    rp = np.random.permutation(wiki_data.__len__())[:batch_size]
    # Get X, y
    batch = [wiki_data.__getitem__(idx) for idx in list(rp)]
    # Return
    return(batch)

# Function to process a batch
def process_batch(batch, device = "cpu"):
    """
    Process a minibatch for handing off to the HAN
    """
    # Get the length of a document in the batch
    doc_len = np.max([b[0].shape[1] for b in batch])
    # Place the first sentences for each doc in one list, second sentences also etc.
#    seq_final = []
#    seq_lens = []
    # Pad documents with fewer sentences than the maximum number of sequences
    # This allows training of documents of different size
    for j in range(len(batch)):
        if batch[j][0].shape[1] < doc_len:
            padder = torch.zeros(1,doc_len).type(torch.long).to(device)
            padder[:, :batch[j][0].shape[1]] = batch[j][0]
            batch[j] = (padder, batch[j][1])
    # Get sequences
    sent_seq = [b[0].to(device) for b in batch]
    # Record lengths of sequences
    sent_lens = [b[0].shape[1] for b in batch]
    # Pad the sequence
    sent_seq_padded = pad_sequence(sent_seq, batch_first=True, padding_value=0).to(device)
    return(sent_seq_padded,sent_lens)

# Function to split input data into train // test
def split_data(X, y, seed = None, p = 0.05):
    """
    Split data into train and test
    """
    # Create batched data
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(X))
    # Get proportion
    num_val = int(np.round(len(X) * p, 0))
    train_idx = indices[:len(X) - num_val]
    test_idx = indices[(len(X) - num_val):]
    # Split
    train_data = [X[index] for index in train_idx]
    train_label = [y[index] for index in train_idx]
    val_label = [y[index] for index in test_idx]
    val_data = [X[index] for index in test_idx]
    # Return
    return((train_data, train_label), (val_data, val_label))

# Training regime for HAN model
def train_lstmn(train_x, train_y, model, optimizer, criterion, epochs = 10, 
              val_split = .1, batch_size=64, device = "cpu"):
    """
    Train a Hierarchical Attention Network

    :param train_x: input documents. Structured as a list of lists, where one entry is a list of input sentences.
                all input sentences must be of the same size.
    :param train_y: numpy array containing the output labels
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
    n_examples = len(train_x)
    # Keep track of training loss / accuracy
    training_loss = []
    training_acc = []
    validation_loss = []
    validation_acc = []
    validation_precision = []
    validation_recall = []
    validation_f1 = []
    # For each epoch, train the mopdel
    for epoch in range(0, epochs):
        epoch += 1
        running_loss = 0.0
        running_acc = 0.0
        # Split data
        batch_train, batch_val = split_data(train_x, train_y, p = val_split)
        # Make datasets
        batch_train_data = WikiDocData(batch_train[0], batch_train[1])
        batch_val_data = WikiDocData(batch_val[0], batch_val[1])
        # For each train/test example
        n_iter = n_examples // batch_size
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
            labels_ground_truth = torch.tensor([b[1] for b in current_batch]).to(device)
            # Zero gradients
            model.zero_grad()
            # Predict output
            predict_out = model(seqs, torch.tensor(lens).type(torch.long).to(device), batch_size)
            # Get max
            predict_class = torch.argmax(predict_out, dim=1).cpu().numpy()
            # Loss
            loss_out = criterion(predict_out, labels_ground_truth)
            # As item
            loss_value = loss_out.cpu().item()
            # GT labels to numpy
            labels_ground_truth = labels_ground_truth.cpu().numpy()
            acc_batch = sum(predict_class == labels_ground_truth) / labels_ground_truth.shape[0]
            # Update loss and accuracy
            running_loss += (loss_value - running_loss) / (i + 1)
            running_acc += (acc_batch - running_acc) / (i + 1)
            # Print if desired
            if i % 5 == 0:
                print("Loss is {} on iteration {}/{} for epoch {} ...".format(np.round(running_loss, 3), i, n_iter, epoch))
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
            io = batcher(batch_val_data, batch_size)
            # Process true label
            ytrue = [doc[1] for doc in io]
            ytrue = torch.tensor(ytrue).to(device)
            # Process batches
            seqs, lens = process_batch(io, device = device)
            # To outcome probabilities
            out = model(seqs, lens)
            loss_out = criterion(out, ytrue)
            # To class labels
            out = torch.argmax(out, dim=1)
        # Make true values into numpy array
        ytrue = ytrue.cpu().numpy()
        # Metrics
        val_metrics = metrics.precision_recall_fscore_support(ytrue,
                                                              out.cpu().numpy(),
                                                              average="weighted")
        # Acc
        val_acc = np.round(sum(out.cpu().numpy() == ytrue) / ytrue.shape[0], 3)
        validation_acc.append(val_acc)
        validation_loss.append(loss_out.cpu().item())
        validation_precision.append(val_metrics[1])
        validation_recall.append(val_metrics[2])
        validation_f1.append(val_metrics[0])
        # Print
        print("-------------")
        print("Training Loss is {} at epoch {} ...".format(np.round(running_loss, 3), epoch))
        print("Training accuracy is {} at epoch {} ...".format(np.round(running_acc, 3), epoch))
        print("Validation accuracy is {} at epoch {} ...".format(val_acc, epoch))
        print("-------------")

    # Return
    return(model, {"training_loss": training_loss,
                   "training_accuracy": training_acc,
                   "validation_loss":validation_loss,
                   "validation_accuracy": validation_acc,
                   "validation_precision":validation_precision,
                   "validation_recall":validation_recall,
                   "validation_f1":validation_f1})

																			  
#%% LSTMN

class LSTMN(nn.Module):
    def __init__(self, weights, batch_size, num_classes, bidirectional=False, nb_lstm_layers=1, nb_lstm_units=32, dropout_prop = 0):
        """
        Implementation of a Long-Short Term Memory Network (LSTMN).

        :param weights: Pre-trained embedding weights
        :batch_size: size of the minibatches passed to the LSTMN.
        :num_classes: number of output classes in the classification task.
		:bidirectional: should the LSTM use bidirectional units.
		:nb_lstm_layers: number of LSTM layers.
		:nb_lstm_units: number of LSTM units per layer.
        """
        super(LSTMN, self).__init__()

        self.nb_lstm_layers = nb_lstm_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = weights.shape
        self.batch_size = batch_size
        self.num_classes = num_classes 
        self._dropout_prop = dropout_prop

        # when the model is bidirectional we double the output dimension
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.bidirectional = bidirectional
        
        # Embedding
        self.embedding = Embedding_FastText(weights, freeze_layer = True)

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim[1], 
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            bidirectional = self.bidirectional
        )

        # output layer which projects back to tag space
        self.hidden_to_label = nn.Linear(self.nb_lstm_units, self.num_classes)
    
    def forward(self, seqs, seq_lens, batch_size=None):
        # Embedding
        embedded = self.embedding(seqs)
        embedded = embedded.permute(2, 0, 3, 1)
        
		# reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        if batch_size is None:
            h_0 = Variable(torch.zeros(self.nb_lstm_layers * self.num_directions, self.batch_size, self.nb_lstm_units).cuda()) # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(self.nb_lstm_layers * self.num_directions, self.batch_size, self.nb_lstm_units).cuda()) # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(self.nb_lstm_layers * self.num_directions, batch_size, self.nb_lstm_units).cuda())
            c_0 = Variable(torch.zeros(self.nb_lstm_layers * self.num_directions, batch_size, self.nb_lstm_units).cuda())
        output, (final_hidden_state, final_cell_state) = self.lstm(torch.squeeze(embedded), (h_0, c_0))
        
        # Apply dropout
        final_hidden_state = F.dropout(final_hidden_state, p=self._dropout_prop)
        # Linear layer
        final_output = self.hidden_to_label(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        # Softmax
        final_output = F.softmax(final_output, dim = 1)
        return final_output