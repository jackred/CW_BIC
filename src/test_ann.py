# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed  this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from ann import ANN
from pso import PSO, maximise, minimise, Rosenbrock
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


def train_ANN_PSO(inputs, res_ex, n_particle, n_iter, nb_h_layers,
                  nb_neurons_layer,
                  min_bound, max_bound, cognitive_weight,
                  social_weight, inertia_start, inertia_end,
                  velocity_max, activation, draw_graph=False):
    nb_neurons = [len(inputs[0])]
    nb_neurons.extend([nb_neurons_layer] * nb_h_layers)
    nb_neurons.append(1)
    print(nb_neurons)
    ann = ANN(nb_neurons=nb_neurons, nb_layers=len(nb_neurons),
              activation=activation)
    dim = sum(nb_neurons[i] * nb_neurons[i+1]
              for i in range(len(nb_neurons)-1)) + len(nb_neurons) - 1
    pso = PSO(dim, lambda params: active(params, ann, inputs, res_ex),
              max_iter=n_iter, n_particle=n_particle, comparator=minimise,
              min_bound=min_bound, max_bound=max_bound,
              cognitive_weight=cognitive_weight, social_weight=social_weight,
              inertia_start=inertia_start, inertia_end=inertia_end,
              velocity_max=velocity_max)
    if draw_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex, dry=True)
    pso.run()
    return pso, ann


def main():
    name = '../Data/1in_tanh.txt'
    inputs, res_ex = read_input(name)
    real_time_graph = False
    args = [1, 3, -7.176582343826539, 3.0666915574121836, 0.0,
            2.2625112213772844, -0.26381961890844063, 1.0, 50.0, ann_help.atan]
    pso, ann = train_ANN_PSO(inputs, res_ex, 40, 120, *args,
                             draw_graph=real_time_graph)
    if not real_time_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex, dry=True)
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


def train_PSO(function, comparator, n_particle, n_iter, min_bound, max_bound,
              cognitive_weight, social_weight, inertia_start, inertia_end,
              velocity_max):
    pso = PSO(function.dimension, function.evaluate, max_iter=n_iter,
              n_particle=n_particle, comparator=comparator,
              min_bound=min_bound, max_bound=max_bound,
              cognitive_weight=cognitive_weight, social_weight=social_weight,
              inertia_start=inertia_start, inertia_end=inertia_end,
              velocity_max=velocity_max, version=2011, endl='\r')
    pso.run()
    print('\n')
    return pso


if __name__ == '__main__':
    main()
    # rosenbrock = Rosenbrock(5)
    # res = []
    # for i in range(30):
    #     pso = train_PSO(rosenbrock, minimise, 40, 2500, -5, 10, 2.161727220149314, 0.8476159691879278, 0.4539167184315849, 0.7073827080382431, 1e-06)
    #     res.append(pso.best_score)
    # res.sort()
    # print(res)
    # mean = sum(res) / len(res)
    # print('mean:', mean)
    # print('median:', res[len(res)//2])
    # print('std:', sqrt((1/(len(res) - 1)) * sum([(i - mean) ** 2 for i in res])))
    # print('best:', res[0])
    # print('worst:', res[-1])
