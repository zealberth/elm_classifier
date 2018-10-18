import numpy as np
from elm_regressor import ELM

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

model = ELM()

model.fit(X_train, y_train, num_neurons=30)

y_hat = model.predict(X_test)

print(mean_squared_error(y_test, y_hat))