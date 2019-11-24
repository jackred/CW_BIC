# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset:4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble


from pso import PSO, minimise
from test_ann import train_ANN_PSO
from train_help import graph_opso, read_input
from pso_json import get_born_config
from ann_help import ACTIVATIONS, scale, MIN_BOUND, MAX_BOUND, Rosenbrock
import matplotlib.pyplot as plt


def scale_args(args, born):
    # Iterate through all arguments to scale them between specific born
    i = 0
    for key in born:
        args[i] = scale(args[i], born[key][0], born[key][1])
        i += 1

    # Round nb_h_layers and nb_neurons_layer to have int values
    args[1] = round(args[1])
    args[2] = round(args[2])

    # Get activation functions
    i_activation = round(scale(args[-1], 0, len(ACTIVATIONS) - 1))
    activations = [ACTIVATIONS[i_activation] for _ in range(args[1] + 1)]
    return args[:-1] + [activations]


def fitness_mean(*args):
    res = []
    best_pso = None
    best_score = float("inf")
    for i in range(4):
        pso, _ = train_ANN_PSO(*args)
        res.append(pso.best_global_score)
        if pso.best_global_score < best_score:
            best_score = pso.best_global_score
            best_pso = pso
    return sum(res) / len(res), best_pso


def train_PSO_PSO_ANN(inputs, res_ex, born, draw_graph=False):
    dim = 11
    opso = PSO(dim,
               lambda param: fitness_mean(inputs, res_ex, 50, 25,
                                          *scale_args(param, born)),
               max_iter=10, n_particle=8, n_neighbor=4,
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
    real_time_graph = False
    born = get_born_config()
    pso = train_PSO_PSO_ANN(inputs, res_ex, born, draw_graph=real_time_graph)
    print(scale_args(pso.best_position, born))
    if not real_time_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex, opso=True)
        pso.draw_graphs()
    plt.show()


if __name__ == '__main__':
    main()
