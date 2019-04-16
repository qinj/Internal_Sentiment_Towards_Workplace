import os
import unittest
import pickle
from src.score import create_train_test_data, logcosh
from src.score import random_forest_model, xgboost_model, neural_network_model
import warnings

warnings.simplefilter("ignore")

class TestScore(unittest.TestCase):
    # testing create_train_test_data
    def test1(self):
        # arrange, act
        X_train, X_test, y_train, y_test = create_train_test_data()

        # assert
        self.assertTrue(X_train.shape[0] == 21142)
        self.assertTrue(X_train.shape[1] == 4015)
        self.assertTrue(y_train.shape[0] == 21142)
        self.assertTrue(y_train.shape[1] == 1)
        self.assertTrue(X_test.shape[0] == 5286)
        self.assertTrue(X_test.shape[1] == 4015)
        self.assertTrue(y_test.shape[0] == 5286)
        self.assertTrue(y_test.shape[1] == 1)

    # testing logcosh
    def test2(self):
        # arrange, act
        res = logcosh(0.2, 0.4)

        # assert
        self.assertTrue(res == 0.01986807184000736)

    # testing models
    def test3(self):
        # arrange
        X_train, X_test, y_train, y_test = create_train_test_data()
        with open('src/models/random_forest_regressor.pkl', 'rb') as f:
            RF_model = pickle.load(f)
        with open('src/models/gradient_boosting_regressor.pkl', 'rb') as f:
            XGB_model = pickle.load(f)
        with open('src/models/rnn_model.pkl', 'rb') as f:
            NN_model = pickle.load(f)

        # act
        rmse_rf, mse_rf, mae_rf, log_cosh_rf = random_forest_model(RF_model, X_test, y_test)
        rmse_xgb, mse_xgb, mae_xgb, log_cosh_xgb = xgboost_model(XGB_model, X_test, y_test)
        rmse_nn, mse_nn, mae_nn, log_cosh_nn = neural_network_model(NN_model, X_test, y_test)

        # assert
        self.assertTrue(rmse_rf == 0.91)
        self.assertTrue(mse_rf == 0.84)
        self.assertTrue(mae_rf == 1.15)
        self.assertTrue(log_cosh_rf == 0.66)

        self.assertTrue(rmse_xgb == 0.85)
        self.assertTrue(mse_xgb == 0.73)
        self.assertTrue(mae_xgb == 1.25)
        self.assertTrue(log_cosh_xgb == 0.75)

        self.assertTrue(rmse_nn == 0.96)
        self.assertTrue(mse_nn == 0.93)
        self.assertTrue(mae_nn == 0.77)
        self.assertTrue(log_cosh_nn == 0.35)


if __name__ == '__main__':
    unittest.main()
