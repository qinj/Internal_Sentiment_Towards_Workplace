from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import filter_tokens, validate_string, extract_bow_from_column

nlp_df = pd.read_csv('../data/df_with_nlp.csv', index_col=0)
X = nlp_df
y = pd.read_csv("../data/work-balance-stars.csv", header=None, index_col=0).values


with open('models/gradient_boosting_regressor.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/vectorizer_pros.pkl', 'rb') as f:
    cv_pros = pickle.load(f)
with open('models/vectorizer_cons.pkl', 'rb') as f:
    cv_cons = pickle.load(f)
with open('models/standardizer.pkl', 'rb') as f:
    sc = pickle.load(f)

app = Flask(__name__, static_url_path="")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a random prediction."""

    data = request.json
    data_df = pd.DataFrame()
    # PROS
    data_df['pros'] = pd.Series(data['pros'])
    data_df['pros'] = data_df['pros'].str.lower()
    data_df['filtered_pros'] = extract_bow_from_column(data_df['pros'])
    corpus_pros = [row for row in data_df['filtered_pros']]
    cv_array_pros = cv_pros.transform(corpus_pros).toarray()
    cv_dict_pros = {}
    for key in cv_pros.vocabulary_:
        cv_dict_pros["word_pro_" + key] = cv_array_pros[:,cv_pros.vocabulary_[key]]
    cv_df_pros = pd.DataFrame(cv_dict_pros, index=[0])

    # CONS
    data_df['cons'] = pd.Series(data['cons'])
    data_df['cons'] = data_df['cons'].str.lower()
    data_df['filtered_cons'] = extract_bow_from_column(data_df['cons'])
    corpus_cons = [row for row in data_df['filtered_cons']]
    cv_array_cons = cv_cons.transform(corpus_cons).toarray()
    cv_dict_cons = {}
    for key in cv_cons.vocabulary_:
        cv_dict_cons["word_con_" + key] = cv_array_cons[:,cv_cons.vocabulary_[key]]
    cv_df_cons = pd.DataFrame(cv_dict_cons, index=[0])

    # Get placeholder for other values
    df = pd.DataFrame({'culture-values-stars': [int(data['culture'])],
                        'comp-benefit-stars': [int(data['benefits'])],
                        'career-opportunities-stars': [int(data['career'])],
                        'senior-management-stars': [int(data['senior'])],
                        })
    df['helpful-count'] = pd.Series([0])
    df['is_current_employee'] = pd.Series([1])
    df['amazon_earnings_this_quarter'] = pd.Series([3.027])
    df['stock_price'] = pd.Series([1501.97])
    df['year'] = pd.Series([2018])
    df['quarter'] = pd.Series([4])
    df['incomplete_review'] = pd.Series([0])
    df['pros_len'] = pd.Series([len(data['pros'])])
    df['cons_len'] = pd.Series([len(data['cons'])])
    non_nlp_df = df[['culture-values-stars', 'career-opportunities-stars',
                       'comp-benefit-stars', 'senior-management-stars', 'helpful-count',
                       'is_current_employee', 'year', 'quarter', 'amazon_earnings_this_quarter', 'stock_price', 'incomplete_review', 'pros_len', 'cons_len']]
    new_df = pd.concat([non_nlp_df, cv_df_pros, cv_df_cons], axis=1)
    new_df['timesteps'] = pd.Series([184])
    new_df['pro-con-len-ratio'] = new_df['pros_len']/new_df['cons_len']

    # Standardize continuous variables
    col_names = ['helpful-count', 'amazon_earnings_this_quarter', 'stock_price', 'pros_len', 'cons_len', 'pro-con-len-ratio']
    features = new_df[col_names]
    features = sc.transform(features.values)
    new_df[col_names] = features
    new_df = new_df.reindex(sorted(new_df.columns), axis=1)
    prediction = model.predict(new_df)
    return jsonify({'Projected Work/Life Balance Score': prediction.tolist()})

@app.route('/')
def index():
   """Return the main page."""

   return render_template('index.html')

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8080, debug=True)
