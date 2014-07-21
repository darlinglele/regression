from numpy import *
import numpy as np
from scaler import *


class LogisticRegressor():

    def __init__(self, alpha=0.1, n_iter=20, scaler=None):
        self.alpha = alpha
        self.n_iter = n_iter
        self.scaler = scaler        

    def fit(self, X, Y):
        self.X = self.scaler.fit_transform(X) if self.scaler else X
        self.Y = Y
        self.W = ones((shape(self.X)[1], 1))
        self.intercept = 0
        for n in xrange(self.n_iter):
            for x, y in zip(self.X, Y):
                update = y - self.sigmod(x * self.W + self.intercept)
                self.intercept += self.alpha * update * 1.
                self.W += self.alpha * \
                    x.transpose() * update

    def predict(self, X):
        X = self.scaler.transform(X) if self.scaler else X
        return [self.sigmod(x * self.W + self.intercept) for x in X]

    def sigmod(self, z):
        return 1. / (1. + exp(-z))

    def likelihood(self):
        l = 1.0
        for x, y in zip(self.X, self.Y):
            l *= self.sigmod(x * self.W) ** int(
                y) * (1 - self.sigmod(x * self.W)) ** (1 - int(y))
        return float(l)
