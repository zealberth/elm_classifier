"""Extreme Learning Machine Regression."""
import numpy as np
import sklearn

from sklearn.base import BaseEstimator, RegressorMixin


class ELM(BaseEstimator, RegressorMixin):
    def __init__(self, m_weights=None,w_weights=None):
        self.m_weights = m_weights # pesos da camada vísivel
        self.w_weights = w_weights # pesos da camada oculta

    def fit(self, x_train, y_train, num_neurons=10):
        x_train = np.c_[-1*np.ones(x_train.shape[0]), x_train]

        self.m_weights = np.random.randn(x_train.shape[1],num_neurons)

        u = x_train @ np.asmatrix(self.m_weights)

        H = 1./(1 + np.exp(-u))

        self.w_weights = np.linalg.pinv(H) * np.asmatrix(y_train).T

        return self

    def predict(self, x_test):
        x_test = np.c_[-1*np.ones(x_test.shape[0]), x_test]

        u = x_test @ np.asmatrix(self.m_weights)

        H = 1./(1 + np.exp(-u))

        return H @ np.asmatrix(self.w_weights)
