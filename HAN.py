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

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
# This is a technical thing
# See stackoverflow:
#   - PyTorch: training with GPU gives worse error than training the same thing with CPU
torch.backends.cudnn.enabled = False

"""
HAN utility functions:
    To make the HAN self-contained, I put all utility functions in this python file. The preprocessing
    steps and dataset construction are a little different from the other models. The preprocessing 
    functions are as follows:
        1. Embedding_FastText: creates a Pytorch embedding layer from pre-trained weights
        2. WikiDocData: Pytorch Dataset used to store & retrieve wikipedia data.
        3. batcher: function that creates minibatches for training
        4. process_batch: processes a minibatch of wikipedia articles.
        5. split_data: function that splits wikipedia data into train & test
        6. train_han: training regime for the HAN.
"""

# Create FastText embedding for PyTorch
def Embedding_FastText(weights, freeze_layer = True):
    """Set up a pytorch embedding matrix"""
    examples, embedding_dim = weights.shape
    # Set up layer
    embedding = nn.Embedding(examples, embedding_dim)
    # Add weights
    embedding.load_state_dict({"weight": weights})
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
        X = [torch.tensor(sent).type(torch.long) for sent in self.X[index]]
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
    doc_len = np.max([len(b[0]) for b in batch])
    # Place the first sentences for each doc in one list, second sentences also etc.
    seq_final = []
    seq_lens = []
    # Pad documents with fewer sentences than the maximum number of sequences
    # This allows training of documents of different size
    for j in range(len(batch)):
        if len(batch[j][0]) < doc_len:
            batch[j] = (batch[j][0] + (doc_len - len(batch[j][0])) * [torch.tensor([0]).type(torch.long)], batch[j][1])
    for i in range(doc_len):
        # Get sequences
        sent_seq = [b[0][i] for b in batch]
        # Record lengths of sequences
        sent_lens = [len(sent) for sent in sent_seq]
        # Create numpy
        # Pad the sequence
        sent_seq_padded = pad_sequence(sent_seq, batch_first=True, padding_value=0).to(device)
        # Append
        seq_final.append(sent_seq_padded)
        seq_lens.append(sent_lens)
    # Return
    return(seq_final, seq_lens)

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
        batch_train, batch_val = split_data(X, y, p = val_split)
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
            labels_ground_truth = torch.tensor([b[1] for b in current_batch]).to(device)
            # Zero gradients
            model.zero_grad()
            # Predict output
            predict_out = model(seqs, torch.tensor(lens).type(torch.long).to(device))
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

"""
PyTorch modules:
    The PyTorch modules below implement the HAN with attention. The following modules are added.
        1. Attention: implements the attention mechanism described in Yang et al.
        2. word_encoder: applies a bidirectional GRU and attention to input sentences.
        3. sentence_encoder: applies a bidirectional GRU and attention to output of the word encoder
        4. HAN: implementation of the Hierarchical Attention Network. Calls modules 1-3.

    I've taken some inspiration from the following existing implementations:
        - https://github.com/uvipen/Hierarchical-attention-networks-pytorch
        - https://github.com/pandeykartikey/Hierarchical-Attention-Network
"""

#%% Attention module

class Attention(nn.Module):
    def __init__(self, hidden_size):
        """
        Attention mechanism.

        :param hidden_size: size of the hidden states of the bidirectional GRU

        :seealso: 
            - https://pytorch.org/docs/stable/nn.html#gru for the output size of the GRU encoder (bidirectional)
            - Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016, June). Hierarchical attention networks 
              for document classification. In Proceedings of the 2016 conference of the North American chapter of the 
              association for computational linguistics: human language technologies (pp. 1480-1489).
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

        :param hidden_states: The hidden states of the input sequence at time T
        
        :return: context vector (weighted GRU output) and attention weights
        """
        # (see equation 5)
        u = torch.tanh(self._layer1(hidden_states))
        # (see equation 6)
        alphas = F.softmax(self._layer2(u), dim=1)
        # --> current dimensions: X x Y x Z
        # Sentence vectors
        # (see equation 7)
        # Apply the attention weights (alphas) to each hidden state
        sentence = torch.sum(torch.mul(alphas, hidden_states), dim=1)
        # Return
        return(sentence, alphas)

#%% Word encoder

class word_encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        """
        Word encoder. This part takes a minibatch of input sentences, applies a GRU and attention
         and returns the sequences.

        :param embedding_size: Size of the word embedding
        :param hidden_size: number of hidden units in the word-level GRU
        """
        super(word_encoder, self).__init__()
        self._hidden_size = hidden_size
        # Word-GRU
        self.GRU = nn.GRU(embedding_size, self._hidden_size,
                        bidirectional=True, batch_first=True)       
        # Attention
        self.attention = Attention(self._hidden_size)
    def forward(self, inputs_embedded):
        """
        :param inputs_embedded: word embeddings of the mini batch at time t (sentence x seq_length)

        :return: tuple containing:
            (1) weighted GRU annotations (GRU output weighted by the attention vector)
            (2) [final hidden state of GRU (unweighted), attention weights]
        """
        # Bidirectional GRU
        output_gru, last_hidden_state = self.GRU(inputs_embedded)
        # Unpack packed sequence
        output_padded, output_lengths = pad_packed_sequence(output_gru, batch_first=True)
        # Attention
        output_attention, att_weights = self.attention(output_padded)
        # Return
        return(output_attention.unsqueeze(dim=0), [last_hidden_state, att_weights])

#%% Sentence encoder

class sentence_encoder(nn.Module):
    def __init__(self, word_hidden_size, hidden_size):
        """
        Sentence encoder. This part takes as its input a minibatch of documents which have been created by
         the word encoder. It applies a GRU, attention and returns the weighted GRU output.

        :param word_hidden_size: The number of hidden units of the word encoder.
        :param hidden_size: The number of hidden units used for the sentence encoder.
        """
        super(sentence_encoder, self).__init__()
        self._hidden_size = hidden_size
        # Sentence-GRU
        self.GRU = nn.GRU(word_hidden_size, self._hidden_size,
                          bidirectional=True, batch_first=True)       
        # Attention
        self.attention = Attention(hidden_size)
    def forward(self, encoder_output):
        """
        :param encoder_output: output of the word encoder.

        :return: weighted annotations created by the sentence GRU
        """
        # Bidirectional GRU
        output_gru, last_hidden_state = self.GRU(encoder_output)
        # Attention
        output_attention, att_weights = self.attention(output_gru)
        # Return
        # (weighted attention vector, hidden states of the sentences)
        return(output_attention.unsqueeze(dim=0), [last_hidden_state, att_weights])

#%% HAN

class HAN(nn.Module):
    def __init__(self, weights, hidden_size_words, hidden_size_sent, batch_size, num_classes, device = "cpu",
                 dropout_prop = 0):
        """
        Implementation of a Hierarhical Attention Network (HAN).

        :param weights: Pre-trained embedding weights
        :param hidden_size_words: number of hidden units for the word encoder.
        :param hidden_size_sent: number of hidden units for the sentence encoder.
        :batch_size: size of the minibatches passed to the HAN.
        :num_classes: number of output classes in the classification task.
        """
        super(HAN, self).__init__()
        self._hidden_size_words = hidden_size_words
        self._hidden_size_sent = hidden_size_sent
        self._embedding_dim = weights.shape
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._dropout_prop = dropout_prop
        # Embedding
        self.embedding = Embedding_FastText(weights, freeze_layer = True)
        # Set up word encoder
        self._word_encoder = word_encoder(self._embedding_dim[1], self._hidden_size_words)
        # Set up sentence encoder
        self._sentence_encoder = sentence_encoder(self._hidden_size_words * 2, self._hidden_size_sent)
        # Set up a linear layer
        self._linear1 = nn.Linear(self._hidden_size_sent * 2, self._num_classes)
    def forward(self, seqs, seq_lens, return_attention_weights = False):
        """
        :param batch_in: list of input documents of size batch_size input document with dim (sentence x seq_length)
        :param return_attention_weights: if True, return attention weights

        :return: tensor of shape (batch_size, num_classes) and, optionally, the attention vectors for the word and sentence encoders.
        """
        # Placeholders
        batched_sentences = None
        hid_sent = None
        # If return attention weights
        if return_attention_weights:
            word_weights = []
            sentence_weights = []
        # For each, do ...
        for seq, seq_len in zip(seqs,seq_lens):
            # Embedding
            embedded = self.embedding(seq)
            # Pack sequences
            x_packed = pack_padded_sequence(embedded, seq_len, batch_first=True, 
                                            enforce_sorted=False)
            # Word encoder
            we_out, hid_state = self._word_encoder(x_packed)
            # Cat sentences together
            if batched_sentences is None:
                batched_sentences = we_out
            else:
                batched_sentences = torch.cat((batched_sentences, we_out), 0)
                # Sentence encoder
                out_sent, hid_sent = self._sentence_encoder(batched_sentences.permute(1,0,2))
            # Cat the attention weights
            if return_attention_weights:
                word_weights.append(hid_state[1].data)
                if hid_sent is not None:
                    sentence_weights.append(hid_sent[1].data)
        # Apply dropout
        out_sent_dropout = F.dropout(out_sent.squeeze(0), p=self._dropout_prop)
        # Linear layer & softmax
        prediction_out = F.softmax(self._linear1(out_sent_dropout), dim = 1)
        # Return
        if return_attention_weights:
            return(prediction_out, [word_weights, sentence_weights])
        else:
            return(prediction_out)

