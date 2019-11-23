# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed  this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from ann import ANN
from pso import PSO, maximise, minimise
import matplotlib.pyplot as plt
import ann_help


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


def train_ANN_PSO(inputs, res_ex, n_particle, n_iter, nb_h_layers,
                  nb_neurons_layer,
                  min_bound, max_bound, cognitive_weight,
                  social_weight, inertia_start, inertia_end,
                  velocity_max, activation):
    nb_neurons = [len(inputs[0])]
    nb_neurons.extend([nb_neurons_layer] * nb_h_layers)
    nb_neurons.append(1)
    print(nb_neurons)
    ann = ANN(nb_neurons=nb_neurons, nb_layers=len(nb_neurons),
              activation=activation)
    dim = sum(nb_neurons[i] * nb_neurons[i+1]
              for i in range(len(nb_neurons)-1)) + len(nb_neurons) - 1
    pso = PSO(dim, lambda params: active(params, ann, inputs, res_ex)[0],
              max_iter=n_iter, n_particle=n_particle, comparator=minimise,
              min_bound=min_bound, max_bound=max_bound,
              cognitive_weight=cognitive_weight, social_weight=social_weight,
              inertia_start=inertia_start, inertia_end=inertia_end,
              velocity_max=velocity_max)
    pso.run()
    return pso, ann


def graph_pso(pso, i, s='PSO'):
    plt.subplot(i)
    plt.title(s + ' ' + "Mean square error evolution")
    plt.plot(pso.best_mean_square_error, color='g', label='Best')
    plt.plot(pso.average_mean_square_error, color='c', label='Average')
    plt.legend()


def graph(pso, ann, res_ex, inputs, dry=False):
    _, res = active(pso.best_position, ann, inputs, res_ex)
    plt.figure(1)
    graph_pso(pso, 211 if dry else 221)
    plt.subplot(212)
    plt.title("Target output and the ANN output comparaison")
    plt.plot([f"{i}: {inputs[i]}" for i in range(len(inputs))], res,
             label='Result')
    plt.plot([f"{i}: {inputs[i]}" for i in range(len(inputs))], res_ex,
             label='Target', linestyle=':')
    plt.tick_params(axis='x', labelrotation=70, width=0.5)
    plt.xticks(range(0, len(inputs), 5))
    plt.legend()
    if dry:
        plt.show()


def graph_all(opso, pso, ann, res_ex, inputs):
    graph(pso, ann, res_ex, inputs)
    graph_pso(opso, 222, 'OPSO')
    plt.show()


if __name__ == '__main__':
    name = '../Data/2in_complex.txt'
    inputs, res_ex = read_input(name)
    nb_h_layers = 3
    nb_neurons_layer = 5
    activation = ann_help.tanh
    min_bound = -5
    max_bound = 5
    pso, ann = train_ANN_PSO(inputs, res_ex, 40, 150, nb_h_layers,
                             nb_neurons_layer,
                             min_bound, max_bound, 2, 2, 0.9, 0.4, 20,
                             activation)
    graph(pso, ann, res_ex, inputs, True)
