# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from random import random
from math import exp, tanh, cos, atan, asinh, sqrt, log, sin
import matplotlib.pyplot as plt


MIN_BOUND = -5
MAX_BOUND = 5

"""
different activation functions
https://en.wikipedia.org/wiki/Activation_function
"""


def default_activation(x):
    return tanh(x)


def heaviside(x):
    return 0 if x < 0 else 1


def sigmoid(x):
    return 1 / (1 + exp(-x))


def identity(x):
    return x


def softsign(x):
    return x / (1 + abs(x))


def relu(x):
    return 0 if x < 0 else x


def softplus(x):
    return log(1+exp(x))


def leaky(x):
    return (0.01 * x) if x < 0 else x


def gaussian(x):
    return exp(-pow(x, 2) / 2)


ACTIVATIONS = [sigmoid, tanh, atan, asinh, softsign]

"""
Help functions
scale: scale a value from an interval to another
mean_square_error
mean_absolute_error
"""


def scale(i, new_min_bound, new_max_bound, min_bound=MIN_BOUND,
          max_bound=MAX_BOUND):
    d = abs(max_bound - min_bound)
    dnew = abs(new_min_bound - new_max_bound)
    res = (((i - min_bound) / d) * dnew) + new_min_bound
    return res


def mean_square_error(d, u):
    return (1 / len(d)) * sum([pow((d[i] - u[i]), 2) for i in range(len(d))])


def mean_absolute_error(d, u):
    return (1 / len(d)) * sum(abs(d[i] - u[i])for i in range(len(d)))


def read_input(name):
    with open(name) as f:
        inputs = []
        res_ex = []
        for i in f:
            tmp = ([float(j) for j in i.split()])
            inputs.append([x for x in tmp[:-1]])
            res_ex.append(tmp[-1])
    return inputs, res_ex


"""
Graph function to display training of pso information
"""


def graph_pso(pso, i, s='PSO'):
    plt.subplot(i)
    plt.title(s + ' ' + "Mean square error evolution")
    plt.plot(pso.best_mean_square_error, color='g', label='Best')
    plt.plot(pso.average_mean_square_error, color='c', label='Average')
    plt.legend()


def graph_opso(pso, opso):
    plt.figure(1)
    graph_pso(pso, 211)
    graph_pso(opso, 212)
    plt.show()


"""
Rosenbrock function for optimisation test
"""


class Rosenbrock:
    def __init__(self, dimension=2):
        self.max_bound = 10
        self.min_bound = -5
        self.dimension = dimension

    def generate_random(self):
        return [random.uniform(self.min_bound, self.max_bound)
                for _ in range(self.dimension)]

    def evaluate(self, xx):
        d = len(xx)
        int_sum = 0
        for i in range(d-1):
            xi = xx[i]
            xnext = xx[i+1]
            new = 100*(xnext-xi**2)**2 + (xi-1)**2
            int_sum = int_sum + new
        y = int_sum
        return y
