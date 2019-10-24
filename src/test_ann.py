# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from ann import ANN

if __name__ == '__main__':
    ann = ANN(nb_neurons=[2, 4, 3], nb_layers=3)
    with open('../Data/1in_cubic.txt') as f:
        for i in f:
            inputs = [float(j) for j in i.split()]
            print(inputs, ann.activation(inputs))
