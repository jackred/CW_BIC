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
            inputs.append([x for x in tmp[:-1]])
            res_ex.append(tmp[-1])
    return inputs, res_ex


def active(weights, ann, inputs, res_ex):
    ann.update_weights(weights)
    res = []
    for i in range(len(inputs)):
        estimation = ann.activation(inputs[i])
        res.append(estimation[0])
    return mean_square_error(res_ex, res), res


if __name__ == '__main__':
    inputs, res_ex = read_input('../Data/2in_complex.txt')
    nb_neurons = [2, 4, 4, 4, 1]
    ann = ANN(nb_neurons=nb_neurons, nb_layers=len(nb_neurons))
    dim = sum(nb_neurons[i] * nb_neurons[i+1] for i in range(len(nb_neurons)-1))
    pso = PSO(dim, lambda weigths: active(weigths, ann, inputs, res_ex)[0],
              400, minimise, min_bound=-10, max_bound=10)
    score, position = pso.run()
    bscore, res = active(position, ann, inputs, res_ex)
    for i in range(len(res)):
        print(res[i], '->', res_ex[i])
    print(score, bscore)
    print(position)
    fig, ax = plt.subplots()
    ax.plot([f"{i}: {inputs[i]}" for i in range(len(inputs))], res, label='Result')
    ax.plot([f"{i}: {inputs[i]}" for i in range(len(inputs))], res_ex, label='Target', linestyle='--')
    ax.tick_params(axis='x', labelrotation=70, width=0.5)
    ax.xaxis.set_ticks(range(0, len(inputs), 3))
    leg = ax.legend()
    plt.show()
