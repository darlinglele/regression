import matplotlib.pyplot as plt
import random
from numpy import *
from scaler import *


class SGDRegressor(object):

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
                update = self.formula(x) - y
                self.intercept -= self.alpha * update * 1.
                self.W -= self.alpha * \
                    x.transpose() * update

    def predict(self, X):
        X = self.scaler.transform(X) if self.scaler else X
        return [float(x * self.W + self.intercept) for x in X]

    def formula(self, x):
        return x * self.W + self.intercept

    def cost(self, X, Y):
        X = self.scaler.transform(X) if self.scaler else X
        return sum([float(x * self.W + self.intercept - float(y)) ** 2 for x, y in zip(X, Y)]) / len(X)
