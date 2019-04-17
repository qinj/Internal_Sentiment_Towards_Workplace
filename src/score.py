import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from pandas.plotting import scatter_matrix
import pickle

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from keras.callbacks import ModelCheckpoint

# change this to which company you want
company = "google"

def create_train_test_data():
    nlp_df = pd.read_csv('../data/df_with_nlp_' + company + '.csv', index_col=0)
    X = nlp_df
    y = pd.read_csv("../data/work-balance-stars_" + company + ".csv", header=None, index_col=0).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = X_train.reindex(sorted(X.columns), axis=1)
    X_test = X_test.reindex(sorted(X.columns), axis=1)
    return X_train, X_test, y_train, y_test

def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.mean(loss)

def random_forest_model(RF_model, X_test, y_test):
    y_pred = RF_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = np.mean(np.abs(y_test - y_pred))
    log_cosh = logcosh(y_test, y_pred)
    return round(rmse, 2), round(mse, 2), round(mae, 2), round(log_cosh, 2)

def xgboost_model(XGB_model, X_test, y_test):
    y_pred = XGB_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = np.mean(np.abs(y_test - y_pred))
    log_cosh = logcosh(y_test, y_pred)
    return round(rmse, 2), round(mse, 2), round(mae, 2), round(log_cosh, 2)

def neural_network_model(NN_model, X_test, y_test):
    X_test = X_test[['culture-values-stars', 'career-opportunities-stars', 'comp-benefit-stars', 'senior-management-stars',
                   'helpful-count', 'is_current_employee', 'year', 'quarter', 'earnings_this_quarter','timesteps']]
    y_pred = NN_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = np.mean(np.abs(y_test - y_pred))
    log_cosh = logcosh(y_test, y_pred)
    return round(rmse, 2), round(mse, 2), round(mae, 2), round(log_cosh, 2)

def main():
    with open('models/random_forest_regressor_' + company + '.pkl', 'rb') as f:
        RF_model = pickle.load(f)

    with open('models/gradient_boosting_regressor_' + company + '.pkl', 'rb') as f:
        XGB_model = pickle.load(f)

    # with open('models/rnn_model.pkl_' + company + '', 'rb') as f:
    #     NN_model = pickle.load(f)

    X_train, X_test, y_train, y_test = create_train_test_data()
    rmse_rf, mse_rf, mae_rf, log_cosh_rf = random_forest_model(RF_model, X_test, y_test)
    rmse_xgb, mse_xgb, mae_xgb, log_cosh_xgb = xgboost_model(XGB_model, X_test, y_test)
    # rmse_nn, mse_nn, mae_nn, log_cosh_nn = neural_network_model(NN_model, X_test, y_test)
    print("Random Forest Regressor Scores")
    print(rmse_rf, mse_rf, mae_rf, log_cosh_rf)
    print("XGBoost Regressor Scores")
    print(rmse_xgb, mse_xgb, mae_xgb, log_cosh_xgb)
    # print("Neural Network Scores")
    # print(rmse_nn, mse_nn, mae_nn, log_cosh_nn)

if __name__ == '__main__':
    main()
