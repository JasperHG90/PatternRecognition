# Text preprocess functions
import contractions
import re
import unicodedata
import os
# Ratio stopwords to special characters
import numpy as np
# Load stopwords
from nltk.corpus import stopwords as sw
swEng = set(sw.words("english"))

# Settings for cleaning text
THRESHOLD_SPECIAL_CHARACTERS = .15
THRESHOLD_STOPWORDS_UPPER = .16
THRESHOLD_STOPWORDS_LOWER = .01
THRESHOLD_DIGITS = .18

# Remove everything between parentheses
def clear_parentheses(string):
    """Remove everything between parentheses"""
    return(re.sub(r'\(.*\)', '', string))

def remove_citations(string):
    """Remove citations"""
    return(re.sub(r'\[[0-9]{1,3}\]', '', string))

# For each text, check ratio of stopwords to special characters
def categorize_characters(string):
    """Count by character type"""
    alpha = digit = special = 0
    for i in range(len(string)):
        if string[i] == " ": continue
        # Count
        if string[i].isalpha():
            alpha = alpha + 1
        elif string[i].isdigit():
            digit = digit + 1
        else:
            special = special + 1
    # Return
    return(special, digit)

def count_stopwords(string):
    """Count stopwords"""
    tokens = string.split(" ")
    stopword_count = 0
    # Check if in stopwords
    for token in tokens:
        if token.strip() in swEng:
            stopword_count += 1
    # Return
    return(stopword_count)

def filter_criterion(string, minimum_characters = 10):
    """Calculate ratio to stopwords"""
    if len(string) < minimum_characters: return((None, None, None))
    # Snippet length
    sl = len(string)
    # Count # stopwords
    swcount = count_stopwords(string)
    # Count special characters
    sccount, dgcount = categorize_characters(string)
    # Return counts
    return( (np.round(sccount / sl, 5), np.round(swcount / sl, 5), np.round(dgcount / sl, 5)) )

def tokens_lower(tokens):
    """All tokens to lower-case"""
    tokens_rep = []
    for token in tokens:
        tokens_rep.append(token.lower())
    return tokens_rep

def remove_punctuation(tokens):
    """Remove punctuation from a token"""
    tokens_rep = []
    for token in tokens:
        tokens_rep.append(re.sub(r'[^\w\s]', '', token))
    return tokens_rep

def replace_contractions(text):
    """Replace contractions in string of text"""
    text = contractions.fix(text)
    # Remove specifically 's
    # Specific contraction that is ignored by contractions library
    return(text.replace("'s", ""))

def remove_non_ascii(tokens):
    """Remove non-asci characters"""
    tokens_rep = []
    for token in tokens:
        tokens_rep.append(unicodedata.normalize('NFKD', token).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
    return(tokens_rep)

def remove_digits(tokens):
    """Remove digits from a list of tokens"""
    tokens_rep = []
    for token in tokens:
        tokens_rep.append(re.sub(r"\d", "#", token))
    return(tokens_rep)

# Define tokenizer function
def tokenize_text(input, lower_tokens = True, remove_digits_token=True):

    """Parse & tokenize a piece of text"""

    # Replace contractions
    try:
        input = replace_contractions(input)
    except IndexError as e:
        print("Document threw index error ... continuing without removing contractions")

    # Remove parentheses and citations
    input = clear_parentheses(input)
    input = remove_citations(input)

    # Check quality of input string
    filter_text = filter_criterion(input)
    # If any is None, then raise error
    if None in filter_text:
        print("String contains None or is too short ...")
        return(None)

    # Tokenize
    tokens = input.split()
    input = " ".join(tokens)

    # Else, check against values
    if filter_text[0] > THRESHOLD_SPECIAL_CHARACTERS: return(None)
    if (filter_text[1] > THRESHOLD_STOPWORDS_UPPER) or (filter_text[1] < THRESHOLD_STOPWORDS_LOWER): return(None)
    if filter_text[2] > THRESHOLD_DIGITS: return(None)

    # Replace non-ascii
    tokens = remove_non_ascii(tokens)

    # Remove punctuation
    # This is also done by keras tokenizer
    #tokens = remove_punctuation(tokens)

    # Remove digits
    if remove_digits_token:
        tokens = remove_digits(tokens)

    # To lowercase
    if lower_tokens:
        tokens = tokens_lower(tokens)

    # If short sentence, then pass
    if len(tokens) <= 4:
        return None

    # Return
    return(" ".join(tokens).strip())

# Load wikipedia data (only first paragraph)

def load_snippets(file):

    """Load the wikipedia snippets pre-processed by Cortical"""

    assert os.path.exists(file), "File '{}' does not exist ...".format(file)

    count = 0
    pages = dict()
    with open(file, "r") as inFile:
        for line in inFile:
            # If 0, then start of a page
            unjoin = line.split("\t")
            # Split on meta information
            meta = unjoin[0].split("|")
            # Get snippet number ==> if this is 0 then start of new page
            snippet_number = int(meta[-1])
            # Filter first pages
            if (snippet_number == 0):
                pages[meta[0]] = unjoin[-1]
                count += 1

    # Return
    return(pages)
