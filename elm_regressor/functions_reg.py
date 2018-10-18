import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

def RegENN(X, y, alpha, n_neighbors=4):
	m = X.shape[0]
	for i in range(m):
		nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X) 
		_, idxs = nbrs.kneighbors(X[i,:]) #idxs para montar o conjunto S

		neigh = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y) 
		y_hat = neigh.predict(X[i,:])

		theta = alpha * np.std(y[idxs[1:]])

		if (y[i] - y_hat) > theta:
			np.delete(X, i, 0)
			np.delete(y, i, 0)
	return X

def RegCNN(X, y, alpha, n_neighbors=1):
	m = X.shape[0]
	X_P = X[0,:]
	y_P = y[0,:]
	temp_X = X.copy()
	temp_y = y.copy()
	for i in range(1,m):
		nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X_P)
		_, idxs = nbrs.kneighbors(temp_X[i,:]) #idxs para montar o conjunto S

		neigh = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y) 
		y_hat = neigh.predict(temp_X[i,:])

		theta = alpha * np.std(y_P[idxs])

		if (y[i] - y_hat) > theta:
			X_P = np.vstack((X_P, temp_X[i,:]))
			y_P = np.vstack((y_P, temp_y[i,:]))

			idx = np.where((X == temp_X[i,:]).all(axis=1))
			np.delete(X, idx, 0)
			np.delete(y, idx, 0)
	return X_P, y_P


from sklearn import datasets
from sklearn.preprocessing import StandardScaler


boston = datasets.load_boston()

X = boston.data
y = boston.target

scaler = StandardScaler().fit(X)
X = scaler.transform(X)

X_selected, y_selected = RegCNN(X, y, alpha=0.5)

from sklearn.neighbors import KNeighborsRegressor
knnr1 = KNeighborsRegressor().fit(X_selected, y_selected)

knnr1.score(X,y)
