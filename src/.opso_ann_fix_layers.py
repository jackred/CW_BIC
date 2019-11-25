# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset:4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble


from pso import PSO, minimise
from pso_ann import train_ANN_PSO
from train_help import graph_opso, read_input
from pso_json import decode_args
from train_help import ACTIVATIONS, scale, MIN_BOUND, MAX_BOUND, Rosenbrock
import matplotlib.pyplot as plt

N_H_LAYER = 3

def scale_args(args, boundary):
    # Iterate through all arguments to scale them between specific born
    i = 0
    for key in boundary:
        if key != "nb_neurons_layer":
            args[i] = scale(args[i], boundary[key][0], boundary[key][1])
            i += 1

    # Scale nb_neurons_layer
    neurons = []
    for j in range(i, i + N_H_LAYER):
        nb_neurons_boundary = boundary["nb_neurons_layer"]
        neurons.append(round(scale(args[j], nb_neurons_boundary[0],
                                   nb_neurons_boundary[1])))

    # Get activation functions
    activations = []
    for j in range(i + N_H_LAYER, i + N_H_LAYER * 2 + 1):
        i_activation = round(scale(args[j], 0, len(ACTIVATIONS) - 1))
        activations.append(ACTIVATIONS[i_activation])

    idx = N_H_LAYER + 1
    return args[:1] + [N_H_LAYER] + [neurons] + args[idx:-idx] + [activations]


def fitness_mean(*args):
    res = []
    best_pso = None
    best_score = float("inf")
    for i in range(2):
        pso, _ = train_ANN_PSO(*args)
        res.append(pso.best_global_score)
        if pso.best_global_score < best_score:
            best_score = pso.best_global_score
            best_pso = pso
    return sum(res) / len(res), best_pso


def train_PSO_PSO_ANN(inputs, res_ex, boundary, draw_graph=False):
    dim = 9 + N_H_LAYER * 2
    opso = PSO(dim,
               lambda param: fitness_mean(inputs, res_ex, 50, 25,
                                          *scale_args(param, boundary)),
               max_iter=20, n_particle=10, n_neighbor=4,
               inertia_start=0.5, inertia_end=0.5, comparator=minimise,
               min_bound=MIN_BOUND, max_bound=MAX_BOUND)
    print("\nRunning...\n")
    if draw_graph:
        opso.set_graph_config(inputs=inputs, res_ex=res_ex, opso=True)
    opso.run()
    return opso


def main():
    name = '../Data/1in_cubic.txt'
    inputs, res_ex = read_input(name)
    real_time_graph = True
    boundary = decode_args('opso', 'boundary', 1)
    pso = train_PSO_PSO_ANN(inputs, res_ex, boundary,
                            draw_graph=real_time_graph)
    print(scale_args(pso.best_position, boundary))
    if not real_time_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex, opso=True)
        pso.draw_graphs()
    plt.show()


if __name__ == '__main__':
    main()
