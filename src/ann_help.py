# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from random import random
from math import exp, tanh, cos, atan


def default_activation(x):
    return tanh(x)


def heaviside(x):
    return 0 if x < 0 else 1


def sigmoid(x):
    return 1 / (1 + exp(-x))


def gaussian(x):
    return exp(-pow(x, 2) / 2)
