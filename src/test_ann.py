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
import math


def mean_square_error(d, u):
    return (1 / len(d)) * sum([pow((d[i] - u[i]), 2) for i in range(len(d))])


def read_input(name):
    with open(name) as f:
        inputs = []
        res_ex = []
        for i in f:
            tmp = ([float(j) for j in i.split()])
            inputs.append([x for x in tmp[:-1]])
            res_ex.append(tmp[-1])
    return inputs, res_ex


def active(params, ann, inputs, res_ex):
    ann.update_params(params)
    res = []
    for i in range(len(inputs)):
        estimation = ann.activation(inputs[i])
        res.append(estimation[0])
    return mean_square_error(res_ex, res), res


if __name__ == '__main__':
    inputs, res_ex = read_input('../Data/1in_cubic.txt')
    nb_neurons = [len(inputs[0]), 6, 1]
    ann = ANN(nb_neurons=nb_neurons, nb_layers=len(nb_neurons))
    dim = sum(nb_neurons[i] * nb_neurons[i+1]
              for i in range(len(nb_neurons)-1)) + len(nb_neurons) - 1
    pso = PSO(dim, lambda weigths: active(weigths, ann, inputs, res_ex)[0],
              500, minimise, min_bound=-5, max_bound=5)
    score, position = pso.run()
    bscore, res = active(position, ann, inputs, res_ex)
    # for i in range(len(res)):
    #     print(res[i], '->', res_ex[i])
    # print(score, bscore)
    # print(position)
    plt.figure(1)
    plt.subplot(211)
    plt.title("Mean square error evolution")
    plt.plot(pso.best_mean_square_error, color='g', label='Best')
    plt.plot(pso.average_mean_square_error, color='c', label='Average')
    plt.legend()
    plt.subplot(212)
    plt.title("Target output and the ANN output comparaison")
    plt.plot([f"{i}: {inputs[i]}" for i in range(len(inputs))], res,
            label='Result')
    plt.plot([f"{i}: {inputs[i]}" for i in range(len(inputs))], res_ex,
            label='Target', linestyle=':')
    plt.tick_params(axis='x', labelrotation=70, width=0.5)
    plt.xticks(range(0, len(inputs), 5))
    plt.legend()
    plt.show()
