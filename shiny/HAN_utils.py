#%% Compute attention weights

from HAN import HAN, WikiDocData, predict_HAN
from argparse import Namespace
import pickle
import numpy as np
import os
import torch
from torch import optim
import torch.nn as nn
from sklearn import metrics
import itertools
import html

#%% Classification pipeline
class ClassificationPipeline:
  def __init__(self, embedding_path, model_weights_path, class_weights_path, tokenizer_path, device = "cpu"):
    self._embedding = embedding_path
    self._model_weights = model_weights_path
    self._class_weights = class_weights_path
    self._device = device
    self._tokenizer_path = tokenizer_path
    # Best parameters
    self.best = Namespace(
        dropout_prop = 0.165,
        hidden_size = 64,
        use_class_weights = True,
        batch_size = 128,
        num_classes = 11,
        learning_rate = 0.007472,
        epochs = 9
    )
    # Set up the model
    self.init_model()
    
  def init_model(self):
    # Read embeddings
    print("here1")
    with open(self._embedding, "rb") as inFile:
      FTEMB = torch.tensor(pickle.load(inFile)).to(self._device)
    # Read class weights
    with open(self._class_weights, "rb") as inFile:
      cw = torch.tensor(pickle.load(inFile)).to(self._device)
    # Load tokenizer
    with open(self._tokenizer_path, "rb") as inFile:
      tokenizer = pickle.load(inFile)
      self._idx_to_word = {v:k for k,v in tokenizer.word_index.items()}
    print("here2")
    # Set up a HAN model
    self._HAN = HAN(FTEMB, self.best.hidden_size, self.best.hidden_size, 
                    self.best.batch_size, self.best.num_classes, 
                    dropout_prop=self.best.dropout_prop)
    self._HAN.to(self._device)
    # Optimizer
    self._optimizer = optim.Adam(self._HAN.parameters(), lr= self.best.learning_rate)
    self._criterion = nn.CrossEntropyLoss(weight=cw)
    # Load state parameters
    print("here3")
    self._HAN.load_state_dict(torch.load(self._model_weights, map_location=torch.device(self._device)))
    
  def predict(self, inputs):
    # Preprocess
    wdd = WikiDocData(inputs, np.zeros(1))
    criterion = self._criterion
    # Predict
    pred_class, attn, _ = predict_HAN(self._HAN, wdd, batch_size=1, return_attention = True)
    # Word/Sentence attention
    word_attn, sent_attn = attn
    # Compute attention weights
    word_weights_by_sentence = []
    word_weights_original = []
    sa = sentence_attention(sent_attn[-1][0,:,:].numpy())
    # Weight the word attention weights by the sentence weights
    for sentence_idx in range(0, len(word_attn)):
        # Subset attention vector
        attv = word_attn[sentence_idx].numpy()
        # Get the vectorized sequence ('sentence')
        seq = inputs[0][sentence_idx]
        # Compute the attention weights
        att_weights = word_attention(attv[0,:,:], seq, self._idx_to_word)
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
    ww = [make_word_weights(weight) for weight in normed_weights]
    # Return
    return(pred_class, ".<br>".join(ww))
    

def word_attention(attention_vector, seq, idx_to_word):
    """
    Compute attention weights for each word in the sentence
    
    :param attention_vector: tensor of shape (sentence_length, word_hidden_dim)
    :param seq: the vectorized sequence of words
    :param idx_to_word: dict that maps sequence integers to words
    
    :return: dictionary where keys are the words in the sequence and value is the attention weight
    """
    # Sequence length
    seq = np.array(seq)
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
