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
from math import sqrt


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


def active(params, ann, inputs, res_ex):
    ann.update_params(params)
    res = []
    for i in range(len(inputs)):
        estimation = ann.activation(inputs[i])
        res.append(estimation[0])
    return mean_square_error(res_ex, res), res


def train_ANN_PSO(inputs, res_ex, n_iter, n_particle, n_neighbor, nb_h_layers,
                  nb_neurons_layer, min_bound, max_bound, cognitive_trust,
                  social_trust, inertia_start, inertia_end,
                  velocity_max, activation, draw_graph=False):
    nb_neurons = [len(inputs[0])]
    nb_neurons.extend([nb_neurons_layer] * nb_h_layers)
    nb_neurons.append(1)
    print(nb_neurons, n_neighbor, activation)
    ann = ANN(nb_neurons=nb_neurons, nb_layers=len(nb_neurons),
              activations=activation)
    dim = sum(nb_neurons[i] * nb_neurons[i+1]
              for i in range(len(nb_neurons)-1)) + len(nb_neurons) - 1
    pso = PSO(dim, lambda params: active(params, ann, inputs, res_ex),
              max_iter=n_iter, n_particle=n_particle,  n_neighbor=n_neighbor,
              comparator=minimise,
              min_bound=min_bound, max_bound=max_bound,
              cognitive_trust=cognitive_trust, social_trust=social_trust,
              inertia_start=inertia_start, inertia_end=inertia_end,
              velocity_max=velocity_max)
    if draw_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex)
    pso.run()
    return pso, ann


def main():
    name = '../Data/1in_sine.txt'
    inputs, res_ex = read_input(name)
    real_time_graph = False
    args = [2, 1, 3, -7.176582343826539, 3.0666915574121836, 0.0,
            2.2625112213772844, -0.26381961890844063, 1.0, 50.0, ann_help.atan]
    args = [2, 2, 5, -3.0, 0.0, 8.0, 3.302816887455289, 2.0, -1.214638225596609,
            38.52018177792397, [ann_help.sigmoid, ann_help.sigmoid, ann_help.sigmoid]]
    pso, ann = train_ANN_PSO(inputs, res_ex, 300, 60, *args,
                             draw_graph=real_time_graph)
    if not real_time_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex)
        pso.draw_graphs()
    plt.show()


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


def train_PSO(function, comparator,  n_iter, n_particle, n_neighbor, min_bound,
              max_bound,
              cognitive_trust, social_trust, inertia_start, inertia_end,
              velocity_max):
    pso = PSO(function.dimension, function.evaluate, max_iter=n_iter,
              n_particle=n_particle, n_neighbor=n_neighbor, comparator=comparator,
              min_bound=min_bound, max_bound=max_bound,
              cognitive_trust=cognitive_trust, social_trust=social_trust,
              inertia_start=inertia_start, inertia_end=inertia_end,
              velocity_max=velocity_max, version=2011, endl='\r')
    pso.run()
    print('\n')
    return pso


if __name__ == '__main__':
    main()
    # rosenbrock = ann_help.Rosenbrock(12)
    # res = []
    # for i in range(3):
    #     pso = train_PSO(rosenbrock, minimise, 2000, 20, 2, -5, 10, 1.8599977724940657, 2.558891680474429, -0.4525474302296826, -0.14651135431992146, 29.323357725071013)
    #     res.append(pso.best_global_score)
    #     print(pso.best_position)
    # res.sort()
    # print(res)
    # mean = sum(res) / len(res)
    # print('mean:', mean)
    # print('median:', res[len(res)//2])
    # print('std:', sqrt((1/(len(res) - 1)) * sum([(i - mean) ** 2 for i in res])))
    # print('best:', res[0])
    # print('worst:', res[-1])
