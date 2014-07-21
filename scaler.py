from numpy import *


class MinMaxScaler():

    def __init__(self):
        self.min = None
        self.max = None

    def fit_transform(self, X):
        self.max = mat([float(max(X[:, i])) for i in xrange(shape(X)[1])])
        self.min = mat([float(min(X[:, i])) for i in xrange(shape(X)[1])])
        return self.transform(X)

    def transform(self, X):
        X_ = X.copy()
        m, n = shape(X_)
        V = self.max - self.min

        for i in xrange(m):
            for j in xrange(n):
                if V[0, j] == 0.:
                    X_[i, j] = 0.
                else:
                    X_[i, j] = (X_[i, j] - self.min[0, j]) / V[0, j]
        return X_
