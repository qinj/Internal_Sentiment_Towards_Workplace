import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import string
import unicodedata

from scipy import sparse
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from nltk import pos_tag
from nltk import RegexpParser
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

import pickle

def filter_tokens(sent):
    stopwords_ = set(stopwords.words('english'))
    punctuation_ = set(string.punctuation)

    return([w for w in sent if not w in stopwords_ and not w in punctuation_])

def validate_string(s):
    letter_flag = False
    number_flag = False
    for i in s:
        if i.isalpha():
            letter_flag = True
        if i.isdigit():
            number_flag = True
    return letter_flag and number_flag

def extract_bow_from_column(column):
    numeric = '0123456789.'
    stopwords_ = set(stopwords.words('english'))
    punctuation_ = set(string.punctuation)
    special_char = ['%', '-', '/']
    filtered_column = []
    for input_string in column:
        tokens_list = [sent for sent in map(word_tokenize, sent_tokenize(str(input_string)))]
        tokens_filtered = list(map(filter_tokens, tokens_list))
        # filter it more
        for tokens in tokens_filtered:
            for idx, token in enumerate(tokens):
                if token.isdigit():
                    tokens.pop(idx)
                for char in special_char:
                    if char in token:
                        tokens.pop(idx)
                        tokens.extend(token.split(char))
        for tokens in tokens_filtered:
            for idx, token in enumerate(tokens):
                if validate_string(token):
        #             both number and letter in it
                    for i, c in enumerate(token):
                        if c not in numeric:
                            break
                    number = token[:i]
                    unit = token[i:]
                    tokens.pop(idx)
                    tokens.append(unit)
        # either porter or snowball work for stemming
        stemmer_porter = PorterStemmer()
        tokens_stemporter = [list(map(stemmer_porter.stem, sent)) for sent in tokens_filtered]
        if len(tokens_stemporter) == 0:
            filtered_column.append('')
        else:
            filtered_column.append(' '.join(tokens_stemporter[0]))
    return filtered_column

def create_Xy_csv(filepath):
    """ filepath = ../data/clean_amazon_reviews.csv """
    amazon_df = pd.read_csv(filepath)
    amazon_df['pros'] = amazon_df['pros'].str.lower()
    amazon_df['filtered_pros'] = extract_bow_from_column(amazon_df['pros'])
    corpus_pros = [row for row in amazon_df['filtered_pros']]
    cv = CountVectorizer(max_features=2000)
    cv_array = cv.fit_transform(corpus_pros).toarray()
    cv_dict = {}
    i = 0
    for key in cv.vocabulary_:
        cv_dict["word_" + key] = cv_array[:,i]
        i += 1
    cv_df = pd.DataFrame(cv_dict)
    non_nlp_df = amazon_df[['culture-values-stars', 'career-opportunities-stars',
                       'comp-benefit-stars', 'senior-management-stars', 'helpful-count',
                       'is_current_employee', 'year', 'quarter', 'amazon_earnings_this_quarter']]

    new_df = pd.concat([non_nlp_df, cv_df], axis=1)
    new_df.to_csv("../data/df_with_nlp.csv")
    amazon_df['work-balance-stars'].to_csv("../data/work-balance-stars.csv")
    return
