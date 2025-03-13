import nltk


#nltk.download()
import csv
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from collections import Counter
import os

MAX_LINES = 100
DIR = 'dataset'
def readFileIntoLists(filename):
    #lines = []
    full_filename = os.path.join(DIR, filename)
    with open(full_filename, encoding="utf8") as csvFile:
        positive = []
        negative = []
        reader = csv.reader(csvFile)
        data = list(reader)
        for row in data[1:]:
            if row[1] == '0':
                positive.append(row[0])
            else:
                negative.append(row[0])
        return positive, negative
def createLexicon(pos: list[str], neg: list[str], lemmatizer: WordNetLemmatizer):
    """
    This function should generate a lexicon (list of words) from the provided positive and negative samples.
    - Tokenize words using word_tokenize()
    - Lemmatize them using lemmatizer.lemmatize()
    - Use Counter() to count word occurrences and filter based on frequency criteria
    """
    pos_token=word_tokenize(pos)
    neg_token=word_tokenize(neg)
    
    #for i in range(pos_token):

    #lemmatizer.lemmatize() 
def sampleHandling(sample: list[str], lexicon, classification, lemmatizer: WordNetLemmatizer):
    """
    This function should convert a given sample into a feature set using the provided lexicon.
    - Tokenize and lemmatize words
    - Create a feature vector where each word in the lexicon is represented as a frequency count
    - Append classification labels to the feature vector
    """
    # Write your code here
    pass
def processData(filename):
    pos, neg = readFileIntoLists(filename)
    lemmatizer = WordNetLemmatizer()
    lexicon = createLexicon(pos, neg, lemmatizer)
    features = []
    features += sampleHandling(pos, lexicon, [1, 0], lemmatizer)
    features += sampleHandling(neg, lexicon, [0, 1], lemmatizer)
    random.shuffle(features)
    return features

if __name__ == '__main__':
    processData('Train.csv')