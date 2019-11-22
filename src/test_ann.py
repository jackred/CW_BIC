# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from ann import ANN
from pso import PSO, maximise, minimise
import matplotlib.pyplot as plt


def mean_square_error(d, u):
    return (1 / len(d)) * sum([pow((d[i] - u[i]), 2) for i in range(len(d))])


def read_input(name):
    with open(name) as f:
        inputs = []
        res_ex = []
        for i in f:
            tmp = ([float(j) for j in i.split()])
            inputs.append([tmp[0]])
            res_ex.append(tmp[1])
    return inputs, res_ex


def active(weights, ann, inputs, res_ex):
    ann.update_weights(weights)
    res = []
    for i in range(len(inputs)):
        estimation = ann.activation(inputs[i])
        res.append(estimation[0])
    return mean_square_error(res_ex, res), res


if __name__ == '__main__':
    inputs, res_ex = read_input('../Data/1in_cubic.txt')
    ann = ANN(nb_neurons=[1, 4, 1], nb_layers=3)
    pso = PSO(8, lambda weigths: active(weigths, ann, inputs, res_ex)[0],
              200, minimise, min_bound=-10, max_bound=10)
    score, position = pso.run()
    bscore, res = active(position, ann, inputs, res_ex)
    for i in range(len(res)):
        print(res[i], '->', res_ex[i])
    print(score, bscore)
    print(position)
    plt.plot(list(range(len(res_ex))), res)
    plt.plot(list(range(len(res_ex))), res_ex)
    plt.show()
