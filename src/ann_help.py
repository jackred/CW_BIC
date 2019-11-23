# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from random import random
from math import exp, tanh, cos, atan, asinh, sqrt, log, sin


MIN_BOUND = -5
MAX_BOUND = 5


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


def scale(i, new_min_bound, new_max_bound, min_bound=MIN_BOUND,
          max_bound=MAX_BOUND):
    d = abs(max_bound - min_bound)
    dnew = abs(new_min_bound - new_max_bound)
    res = (((i - min_bound) / d) * dnew) + new_min_bound
    return res
