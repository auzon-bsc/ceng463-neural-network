import re
import numpy as np

def split_sentence(sen):
    """ Split sentence to tokens """
    if type(sen) == str:
        return sen.split()
    else:
        return None

def only_letters(string):
    """ Trim string so only letters remain """
    return re.sub(r'[^a-zA-Z]', '', string)


def sentence_to_words(sen):
    """" Convert a sentence(str) to words list """
    if type(sen) != str:
        return []
    words = []
    for tkn in split_sentence(sen):
        wrd = only_letters(tkn)
        words.append(wrd)
    return words

def find_unique_words(rows):
    """" Find lowercaseword set of the rows and remove stop words """
    words = set()
    for rw in rows:
        rw_words = sentence_to_words(depure_sentence(str(rw)))
        for wrd in rw_words:
            words.add(wrd.lower())
    with open("stopwords.txt") as file:
        lines = file.readlines()
        stopwords = set([line.rstrip() for line in lines])
    words -= stopwords
    return list(words)

def depure_sentence(sentence):
    """ Clean the sentence from urls, emails, etc. """
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    sentence = url_pattern.sub(r'', sentence)

    # Remove Emails
    sentence = re.sub('\S*@\S*\s?', '', sentence)

    # Remove new line characters
    sentence = re.sub('\s+', ' ', sentence)

    # Remove distracting single quotes
    sentence = re.sub("\'", "", sentence)
        
    return sentence

def construct_input_values(rows, unique_words):
    """" Count the word numbers in rows with respect to unique words """
    input_values = np.zeros((len(rows),len(unique_words)))
    for rw_ind, rw in enumerate(rows):
        rw_words = sentence_to_words(rw)
        for wrd in rw_words:
            try:
                input_values[rw_ind][unique_words.index(wrd)] += 1
            except ValueError:
                continue
    return input_values

