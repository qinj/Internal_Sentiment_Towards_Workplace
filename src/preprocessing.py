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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pickle

################################################################################
#                                                                              #
#                                CLEANING                                      #
#                                                                              #
################################################################################

def preprocessing_clean_glassdoor():
    glassdoor_df = pd.read_csv('../data/google-amazon-facebook-employee-reviews/employee_reviews.csv', index_col=0)
    glassdoor_df.drop(['link'], axis=1, inplace=True)
    employee_titles_df = glassdoor_df["job-title"].str.split(" - ", n = 1, expand = True)
    employee_titles_df.columns = ['Current-Employee', 'role']
    employee_titles_df['is_current_employee'] = employee_titles_df['Current-Employee'].str.contains("Current Employee").astype(int)
    employee_titles_df.drop(['Current-Employee'], axis=1, inplace=True)
    glassdoor_df = pd.concat([glassdoor_df, employee_titles_df], axis=1)
    glassdoor_df['year'] = glassdoor_df['dates'].str.split(', ').str[1]
    for i, row in glassdoor_df.iterrows():
        if row['year'] == '0000' or pd.isnull(row['year']):
            glassdoor_df.drop([i], inplace=True)
    glassdoor_df['month'] = glassdoor_df['dates'].str.split(', ').str[0].str.split(' ').str[1]
    quarter_list = []
    for _, row in glassdoor_df.iterrows():
        if row['month'] == 'Jan' or row['month'] == 'Feb' or row['month'] == 'Mar':
            quarter_list.append(1)
        elif row['month'] == 'Apr' or row['month'] == 'May' or row['month'] == 'Jun':
            quarter_list.append(2)
        elif row['month'] == 'Jul' or row['month'] == 'Aug' or row['month'] == 'Sep':
            quarter_list.append(3)
        elif row['month'] == 'Oct' or row['month'] == 'Nov' or row['month'] == 'Dec':
            quarter_list.append(4)
        else:
            quarter_list.append(5)
    glassdoor_df['quarter'] = quarter_list
    glassdoor_df.columns = ['company', 'location', 'dates', 'job-title', 'summary', 'pros', 'cons',
                            'advice-to-mgmt', 'overall-ratings', 'work-balance-stars',
                            'culture-values-stars', 'career-opportunities-stars',
                            'comp-benefit-stars', 'senior-management-stars', 'helpful-count', 'role', 'is_current_employee', 'year', 'month', 'quarter']
    for i, row in glassdoor_df.iterrows():
        if row['overall-ratings'] == 'none':
            glassdoor_df.loc[i, 'overall-ratings'] = '-1'
        if row['work-balance-stars'] == 'none':
            glassdoor_df.loc[i, 'work-balance-stars'] = '-1'
        if row['culture-values-stars'] == 'none':
            glassdoor_df.loc[i, 'culture-values-stars'] = '-1'
        if row['career-opportunities-stars'] == 'none':
            glassdoor_df.loc[i, 'career-opportunities-stars'] = '-1'
        if row['comp-benefit-stars'] == 'none':
            glassdoor_df.loc[i, 'comp-benefit-stars'] = '-1'
        if row['senior-management-stars'] == 'none':
            glassdoor_df.loc[i, 'senior-management-stars'] = '-1'
    star_list = ['overall-ratings', 'work-balance-stars', 'culture-values-stars', 'career-opportunities-stars', 'comp-benefit-stars', 'senior-management-stars']
    for feature in star_list:
        glassdoor_df[feature] = glassdoor_df[feature].astype('float')
    for i, row in glassdoor_df.iterrows():
        if row['overall-ratings'] == -1:
            glassdoor_df.loc[i, 'overall-ratings'] = None
        if row['work-balance-stars'] == -1:
            glassdoor_df.loc[i, 'work-balance-stars'] = None
        if row['culture-values-stars'] == -1:
            glassdoor_df.loc[i, 'culture-values-stars'] = None
        if row['career-opportunities-stars'] == -1:
            glassdoor_df.loc[i, 'career-opportunities-stars'] = None
        if row['comp-benefit-stars'] == -1:
            glassdoor_df.loc[i, 'comp-benefit-stars'] = None
        if row['senior-management-stars'] == -1:
            glassdoor_df.loc[i, 'senior-management-stars'] = None
    glassdoor_df.to_csv('../data/clean_glassdoor_reviews.csv')
    return

def preprocessing_create_amazon_with_earnings(glassdoor_filepath, amazon_earnings_filepath):
    glassdoor_df = pd.read_csv(glassdoor_filepath)
    q_earnings_df = pd.read_csv(amazon_earnings_filepath)
    amazon_reviews_df = glassdoor_df[glassdoor_df['company'] == 'amazon']
    earnings_list = []
    for _, row in amazon_reviews_df.iterrows():
        net_income = q_earnings_df['Quarterly Net Income (Billions)'].loc[(q_earnings_df['Quarter'] == row['quarter']) & (q_earnings_df['Year'] == row['year'])]
        if len(net_income.values) > 0:
            earnings_list.append(net_income.values[0])
        else:
            earnings_list.append(None)
    amazon_reviews_df['amazon_earnings_this_quarter'] = earnings_list
    overall_mean = np.nanmean(amazon_reviews_df['overall-ratings'])
    work_balance_mean = np.nanmean(amazon_reviews_df['work-balance-stars'])
    culture_mean = np.nanmean(amazon_reviews_df['culture-values-stars'])
    career_mean = np.nanmean(amazon_reviews_df['career-opportunities-stars'])
    benefit_mean = np.nanmean(amazon_reviews_df['comp-benefit-stars'])
    senior_mean = np.nanmean(amazon_reviews_df['senior-management-stars'])
    # standardized
    average_df = pd.DataFrame({'overall': 0, 'work_balance': 0, 'culture': 0, 'career': 0, 'comp_benefits': 0, 'senior': 0}, index=[0])
    average_df.to_csv("../data/average_stars.csv", index=False)

    for i, row in amazon_reviews_df.iterrows():
        if row['overall-ratings'] != row['overall-ratings']:
            amazon_reviews_df.loc[i, 'overall-ratings'] = overall_mean

        if row['work-balance-stars'] != row['work-balance-stars']:
            amazon_reviews_df.loc[i, 'work-balance-stars'] = work_balance_mean

        if row['culture-values-stars'] != row['culture-values-stars']:
            amazon_reviews_df.loc[i, 'culture-values-stars'] = culture_mean

        if row['career-opportunities-stars'] != row['career-opportunities-stars']:
            amazon_reviews_df.loc[i, 'career-opportunities-stars'] = career_mean

        if row['comp-benefit-stars'] != row['comp-benefit-stars']:
            amazon_reviews_df.loc[i, 'comp-benefit-stars'] = benefit_mean

        if row['senior-management-stars'] != row['senior-management-stars']:
            amazon_reviews_df.loc[i, 'senior-management-stars'] = senior_mean
    amazon_reviews_df.to_csv('../data/clean_amazon_reviews.csv')



################################################################################
#                                                                              #
#                                   NLP                                        #
#                                                                              #
################################################################################
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

def preprocessing_nlp(filepath):
    """ filepath = ../data/clean_amazon_reviews.csv
        creates the X and y csv files: df_with_nlp.csv and work-balance-stars.csv
    """
    amazon_df = pd.read_csv(filepath)

    # PROS
    amazon_df['pros'] = amazon_df['pros'].str.lower()
    amazon_df['filtered_pros'] = extract_bow_from_column(amazon_df['pros'])
    corpus_pros = [row for row in amazon_df['filtered_pros']]
    cv_pros = CountVectorizer(max_features=2000)
    cv_array_pros = cv_pros.fit_transform(corpus_pros).toarray()
    cv_dict_pros = {}
    i = 0
    for key in cv_pros.vocabulary_:
        cv_dict_pros["word_pro_" + key] = cv_array_pros[:,cv_pros.vocabulary_[key]]
    cv_df_pros = pd.DataFrame(cv_dict_pros)
    with open('models/vectorizer_pros.pkl', 'wb') as f:
        pickle.dump(cv_pros, f)

    # CONS
    amazon_df['cons'] = amazon_df['cons'].str.lower()
    amazon_df['filtered_cons'] = extract_bow_from_column(amazon_df['cons'])
    corpus_cons = [row for row in amazon_df['filtered_cons']]
    cv_cons = CountVectorizer(max_features=2000)
    cv_array_cons = cv_cons.fit_transform(corpus_cons).toarray()
    cv_dict_cons = {}
    for key in cv_cons.vocabulary_:
        cv_dict_cons["word_con_" + key] = cv_array_cons[:,cv_cons.vocabulary_[key]]
    cv_df_cons = pd.DataFrame(cv_dict_cons)
    with open('models/vectorizer_cons.pkl', 'wb') as f:
        pickle.dump(cv_cons, f)

    non_nlp_df = amazon_df[['culture-values-stars', 'career-opportunities-stars',
                       'comp-benefit-stars', 'senior-management-stars', 'helpful-count',
                       'is_current_employee', 'year', 'quarter', 'amazon_earnings_this_quarter']]

    new_df = pd.concat([non_nlp_df, cv_df_pros, cv_df_cons], axis=1)
    new_df['timesteps'] = (new_df['year'].apply(str).str[2:4] + new_df['quarter'].apply(str))


    scaled_features = new_df.copy()
    col_names = ['culture-values-stars', 'career-opportunities-stars', 'comp-benefit-stars',
                'senior-management-stars', 'helpful-count', 'amazon_earnings_this_quarter']
    features = scaled_features[col_names]
    sc = StandardScaler()
    features = sc.fit_transform(features.values)
    scaled_features[col_names] = features
    scaled_features.sort_values(by=['timesteps'], inplace=True)
    scaled_features.to_csv("../data/df_with_nlp.csv")
    amazon_df['work-balance-stars'].to_csv("../data/work-balance-stars.csv")
    return
