"""Extreme Learning Machine Regression."""
import numpy as np
import sklearn
from scipy.special import expit
from sklearn.base import BaseEstimator, RegressorMixin


class ELM(BaseEstimator, RegressorMixin):
    def __init__(self, num_neurons=10, m_weights=None,w_weights=None):
        self.num_neurons = num_neurons
        self.m_weights = m_weights # pesos da camada vísivel
        self.w_weights = w_weights # pesos da camada oculta

    def fit(self, x_train, y_train):
        x_train = np.c_[-1*np.ones(x_train.shape[0]), x_train]

        self.m_weights = np.random.randn(x_train.shape[1],self.num_neurons)

        u = np.asmatrix(x_train) @ np.asmatrix(self.m_weights)

        H = 1./(1 + expit(-u))

        H = np.c_[-1*np.ones(H.shape[0]), H]

        self.w_weights = np.linalg.lstsq(H, np.asmatrix(y_train).T, rcond=-1)[0]
        
        return self

    def predict(self, x_test):
        x_test = np.c_[-1*np.ones(x_test.shape[0]), x_test]

        u = np.asmatrix(x_test) @ np.asmatrix(self.m_weights)

        H = 1./(1 + expit(-u))

        H = np.c_[-1*np.ones(H.shape[0]), H]

        return np.asmatrix(H) @ np.asmatrix(self.w_weights)
