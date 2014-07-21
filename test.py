from numpy import *
from logistic_regression import *
from scaler import *
from linear_regression import *
import unittest


class LinearRegressorTest(unittest.TestCase):

    def test_linear_regression(self):
        X = mat([[i, i ** 2] for i in linspace(-150, 150, 1000)])
        Y = mat(
            [0.8134 * x[0, 0] + 0.11 * x[0, 1] + 100 for x in X]).transpose()

        test_X = mat([[i, i ** 2] for i in linspace(-150, 150, 130)])
        test_Y = mat(
            [0.8134 * x[0, 0] + 0.11 * x[0, 1] + 100 for x in test_X]).transpose()

        regressor = SGDRegressor(alpha=0.1, n_iter=60, scaler=MinMaxScaler())
        regressor.fit(X, Y)
        cost = regressor.cost(test_X, test_Y)
        if cost > 2:
            raise Exception('the cost ' + str(cost) + ' is to0 much...')
        print 'cost function result:', cost


class LogisticRegressorTest(unittest.TestCase):

    def test_horse_colic(self):

        X = mat([map(lambda x: float(x), x.replace('?', '0').strip().split(' ')[0:-1])
                for x in open('horse.data')])
        Y = mat(
            [int(x.strip().split(' ')[-1]) - 1 for x in open('horse.data')]).transpose()
        regressor = LogisticRegressor(
            alpha=0.02, n_iter=20, scaler=MinMaxScaler())
        regressor.fit(X, Y)

        test_X = mat([map(lambda x: float(x), x.replace('?', '0').strip().split(' ')[0:-1])
                      for x in open('horse.test')])
        test_Y = Y = mat(
            [int(x.strip().split(' ')[-1]) - 1 for x in open('horse.test')]).transpose()

        result = [(float(x), float(y))
                  for x, y in zip(regressor.predict(test_X), test_Y)]

        print 'horse colic predict:', 'error:', len([x for x in result if abs(x[0] - x[1]) >= 0.5]), 'total', len(test_X)

    def test_2_dimensions_dataset(self):

        X = mat([[float(x) for x in line.strip().split('	')[0:-1]]
                for line in open('2dms.data')])
        Y = mat([float(line.strip().split('	')[-1])
                for line in open('2dms.data')]).transpose()

        regressor = LogisticRegressor(
            alpha=0.01, n_iter=40, scaler=MinMaxScaler())
        regressor.fit(X, Y)
        result = [(float(x), float(y))
                  for x, y in zip(regressor.predict(X), Y)]
        print '2_dimesions_dataset:', 'error:', len([x for x in result if abs(x[0] - x[1]) >= 0.5]), 'total:', len(X)

if __name__ == '__main__':
    unittest.main()
