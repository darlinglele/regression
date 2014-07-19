# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import random
import numpy


class SGDRegressor(object):

    def __init__(self, alpha=0.1, n_iter=20, scaling=True):
        self.alpha = alpha
        self.n_iter = n_iter
        self.coefficient = None
        self.intercept = None
        self.scaling = scaling

    def formula(self, coefficient, intercept, X):
        return sum([c * x for c, x in zip(coefficient, X)]) + intercept

    def fit(self, X, Y):
        if self.scaling:
            self.scale(X)
        self.coefficient, self.intercept = [1 for x in xrange(len(X[0]))], 1
        for i in range(self.n_iter):
            for idx, x in enumerate(X):
                y = self.formula(self.coefficient, self.intercept, x)
                self.intercept -= self.alpha * (y - Y[idx]) * 1.
                for j in xrange(len(x)):
                    self.coefficient[j] -= self.alpha * (y - Y[idx]) * x[j]
            if self.cost(X, Y) < 0.0001:
                return self.coefficient, self.intercept
        return self.coefficient, self.intercept

    def scale(self, X):
        for x in xrange(len(X[0])):
            min_x = min([v[x] for v in X])
            max_x = max([v[x] for v in X])
            for v in X:
                v[x] = (v[x] - min_x) / float(max_x - min_x)

    def predict(self, X):
        return [self.formula(self.coefficient, self.intercept, x) for x in X]

    def cost(self, X, Y):
        return sum([(self.formula(self.coefficient, self.intercept, x) - y) ** 2 for x, y in zip(X, Y)]) / (2 * len(X))


if __name__ == '__main__':
	# X has two dimension
    X = [[i, i ** 2] for i in numpy.linspace(-150, 150, 1000)]
    Y = [0.8134 * x[0] + 0.11 * x[1] + 100 for x in X]

    regressor = SGDRegressor(alpha=0.01, n_iter=60)
    regressor.fit(X, Y)
    # Y_ = regressor.predict(X)
    print 'Cost function result:', regressor.cost(X, Y)

