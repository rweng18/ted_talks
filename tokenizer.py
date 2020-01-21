# --------------------------------------------------------------------------- #
# IMPORT PACKAGES
# --------------------------------------------------------------------------- #

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import wordnet
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# --------------------------------------------------------------------------- #
# HELPER FUNCTIONS
# --------------------------------------------------------------------------- #

# Adds in space between punctuation and words
def add_spaces(text):

    # Find instances of punctuation followed by two letters
    cleanr = re.compile('[.!?,;][A-Za-z][A-Za-z]')
    no_spaces = re.findall(cleanr, text)

    if len(no_spaces) > 0:
        for match in no_spaces:
            punc = match[0] # get the punctuation mark
            word = match[1:] # get the start of the word

            if punc != '?':
                text = re.sub(f"{match}", f"{punc} {word}", text)

            # Special case of ?, cannot be escaped
            else:
                text = re.sub(f"[?]{word}", f"? {word}", text)

    return text

# Handles numbers with 1) commas, 2) before hyphens, 3) in places (1st, 2nd, 3rd, etc.)
def handle_numbers(text):

    # Handle numbers with commas
    clean_commas = re.compile('[0-9]+,[0-9][0-9][0-9]')
    num_commas = re.findall(clean_commas, text)

    if len(num_commas) > 0:
        for match in num_commas:
            replace_str = match.replace(',', '')
            text = re.sub(f"{match}", f"{replace_str}", text)

    # Handle numbers with hyphens
    clean_numbers = re.compile('[0-9]+-')
    numbers = re.findall(clean_numbers, text)

    if len(numbers) > 0:
        for match in numbers:
            text = re.sub(f"{match}", "# ", text)

    # Handle 1st, 2nd, 3rd with hyphens
    text = text.replace('1st-', '1st ').replace('2nd-', '2nd ').replace('3rd-', '3rd ')

    # Handle -th with hyphens
    clean_places = re.compile('[0-9]th-')
    places_hyphen = re.findall(clean_places, text)

    if len(places_hyphen) > 0:
        for match in places_hyphen:
            text = re.sub(f"{match}", f"{match[:-1]} ", text)

    return text

def handle_parentheses(text):

    # Add spaces before and after parentheses
    text = text.replace(')', ')\n').replace('(', ' (')

    # Find all parenthetical phrases
    clean_parentheses = re.compile('\(.*\)')
    parentheses = re.findall(clean_parentheses, text)

    if len(parentheses) > 0:
        for match in parentheses:
            try:
                text = re.sub(f'{match}', ' ', text)
            except:
                text = re.sub('(Applause.)', '. ', text)
                text = re.sub('(Laughter.)', '. ', text)
                text = re.sub('(Music.)', '. ', text)
                text = text.replace('(Applause.', '. ')
                text = text.replace('(Laughter.', '. ')
                text = text.replace('(Music.', '. ')
                text = re.sub('\(|\)', '. ', text)

    text = text.replace('( )', '')

    return ' '.join(text.split())

# --------------------------------------------------------------------------- #
# STOP WORDS & PUNCTUATION
# --------------------------------------------------------------------------- #

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

stop_words = list(stop_words)
stop_words.extend(['yeah', 'ya', 'ah', 'um', 'oh', 'actually', 'literally', 'like', 's', 'applause'])

# --------------------------------------------------------------------------- #
# INITIALIZE WORD NET LEMMATIZER
# --------------------------------------------------------------------------- #

lemmatizer = WordNetLemmatizer()

# --------------------------------------------------------------------------- #
# TOKENIZATION FUNCTION
# --------------------------------------------------------------------------- #

def spacy_tokenizer(text):

    # Remove .., ..., ....
    no_ellipses = text.replace('....', '').replace('...', '').replace('..','').replace('…', '')

    # Remove parenthetical phrases
    no_parentheses = handle_parentheses(no_ellipses)

    # Add missing spaces after punctuation
    with_spaces = add_spaces(no_parentheses)

    # Handle numbers with commas
    clean_numbers = handle_numbers(with_spaces)

    # Remove quotation marks
    no_quotes = clean_numbers.replace('\"', ' ').replace('”', ' ').replace('’', '')

    # Address hyphenation issue -- need to revisit
    no_ism = no_quotes.replace('-ism', 'ism')
    no_dash = no_ism.replace('–', ' ').replace('—', ' ').replace('-', '')

    # Remove music notes
    no_notes = no_dash.replace('♪', '').replace('♫', '')

    # SPECIFIC RULE
    no_spec = no_notes.replace('R and D', 'research and development').replace('R & D', 'research and development')

    # Replace all whitespace with one space
    cleantext = ' '.join(no_spec.split())
    cleantext = cleantext.strip()

    # Lemmatize here
    # Creating our token object, which is used to create documents with linguistic annotations.
    # we disabled the parser and ner parts of the pipeline in order to speed up parsing
    mytokens = nlp(cleantext.lower(), disable=['parser', 'ner'])

    # Lemmatizing each token and converting each token into lowercase
    lemmas = []
    for word in mytokens:
        if word.pos_ == 'NOUN':
            lemmas.append(lemmatizer.lemmatize(word.text.lower().strip(), wordnet.NOUN))
        elif word.pos_ == 'VERB':
            lemmas.append(lemmatizer.lemmatize(word.text.lower().strip(), wordnet.VERB))
        elif word.pos_ == 'ADV':
            lemmas.append(lemmatizer.lemmatize(word.text.lower().strip(), wordnet.ADV))
        elif word.pos_ == 'ADJ':
            lemmas.append(lemmatizer.lemmatize(word.text.lower().strip(), wordnet.ADJ))

    lemmas = [word for word in lemmas if word not in stop_words and word not in punctuations]

    # return preprocessed list of tokens
    return lemmas
