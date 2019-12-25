# Preprocess input data

# Load pre-processed wikipedia data
import pickle
import numpy as np
import uuid
import os
import syntok.segmenter as segmenter
from tqdm import tqdm

# Directories
DATA_DIR = "data"

# Files
DATA_PREFIX = "WikiEssentials_L4"

# Tokenization settings
TOKEN_LOWER = False
TOKEN_REMOVE_DIGITS = True # Replace any digit with '#'
MAX_SENT_LENGTH = 8

#%% Read data and preprocess

from preprocess_utils import tokenize_text
import os

inputs = {"History":{}, "Geography":{}, "Philosophy_and_religion":{}, 
          "Mathematics": {}, "Arts": {}, "Technology": {}, "Society_and_social_sciences":{}, 
          "Everyday_life": {}, "Physical_sciences": {}, "People": {},
          "Biology_and_health_sciences": {}}
failed = []
previous_docnr = "default"
#stop = 100
with open(os.path.join(DATA_DIR, "WikiEssentials_L4.txt"), "r") as inFile:
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
                                        lower_tokens=TOKEN_LOWER,
                                        remove_digits_token=TOKEN_REMOVE_DIGITS)
                # If none, pass ...
                if txt_tok is None:
                    failed.append(csent)
                    continue
                else:
                    sent_level.append(csent)
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
# Inspect failed docs
print(len(failed))
# View one
index = 200
failed[index]

# Test
tokenize_text("The economy's of Gujarat is the fifth-largest state economy in India with ₹13.14 lakh crore .",lower_tokens=TOKEN_LOWER,remove_digits_token=TOKEN_REMOVE_DIGITS)

# How many items?
for k, v in inputs.items():
    print(k)
    print(len(v))
print(sum([len(v) for v in inputs.values()]))
    
# Remove people for now
inputs = {k:v for k,v in inputs.items() if k != "People"}

# Save preprocessed data to disk
with open(os.path.join(DATA_DIR, '{}_P3_preprocessed.pickle'.format(DATA_PREFIX)), 'wb') as handle:
    pickle.dump(inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
