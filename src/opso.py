# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset:4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble


from pso import PSO, minimise, Rosenbrock
from test_ann import train_ANN_PSO, read_input, train_PSO, graph_opso
from ann_help import ACTIVATIONS, scale, MIN_BOUND, MAX_BOUND
import matplotlib.pyplot as plt


def active(n_neighbor, nb_h_layers, nb_neurons_layer,
           min_bound, max_bound, cognitive_weight,
           social_weight, inertia_start, inertia_end,
           velocity_max, i_activation):
    n_neighbor = scale(n_neighbor, 2, 16)
    nb_h_layers = round(scale(nb_h_layers, 1, 4))
    nb_neurons_layer = round(scale(nb_neurons_layer, 1, 5))
    min_bound = scale(min_bound, -1, 0)
    max_bound = scale(max_bound, 0, 1)
    cognitive_weight = scale(cognitive_weight, 0, 8)
    social_weight = scale(social_weight, 0, 8)
    inertia_start = scale(inertia_start, -2, 2)
    inertia_end = scale(inertia_end, -2, 2)
    velocity_max = scale(velocity_max, 0.000001, 50)
    i_activation = round(scale(i_activation, 0, len(ACTIVATIONS) - 1))
    res = [n_neighbor, nb_h_layers, nb_neurons_layer,
           min_bound, max_bound, cognitive_weight,
           social_weight, inertia_start, inertia_end,
           velocity_max, ACTIVATIONS[i_activation]]
    return res


def train_mean(*args):
    res = []
    best_pso = None
    best_score = float("inf")
    for i in range(4):
        pso, _ = train_ANN_PSO(*args)
        res.append(pso.best_score)
        if pso.best_score < best_score:
            best_score = pso.best_score
            best_pso = pso
    return sum(res) / len(res), best_pso


def train_PSO_PSO_ANN(inputs, res_ex, draw_graph=False):
    dim = 11
    opso = PSO(dim,
               lambda param: train_mean(inputs, res_ex, 50, 25, *active(*param)),
               max_iter=10, n_particle=8, n_neighbor=4,
               inertia_start=0.5, inertia_end=0.5, comparator=minimise,
               min_bound=MIN_BOUND, max_bound=MAX_BOUND)
    print("\nRunning...\n")
    if draw_graph:
        opso.set_graph_config(inputs=inputs, res_ex=res_ex, dry=False)
    opso.run()
    return opso


def main():
    name = '../Data/1in_cubic.txt'
    inputs, res_ex = read_input(name)
    real_time_graph = True
    pso = train_PSO_PSO_ANN(inputs, res_ex, draw_graph=real_time_graph)
    if not real_time_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex, dry=False)
        pso.draw_graphs()
    plt.show()
    print(active(*pso.best_position))


def train_mean_PSO(*args):
    res = []
    for i in range(4):
        pso = train_PSO(*args)
        res.append(pso.best_score)
    return sum(res) / len(res)


def active_PSO(cognitive_weight, social_weight, inertia_start, inertia_end,
               velocity_max):
    velocity_max = scale(velocity_max, 0.000001, 50)
    social_weight = scale(social_weight, 0, 4)
    cognitive_weight = scale(cognitive_weight, 0, 4)
    inertia_start = scale(inertia_start, -1, 1.2)
    inertia_end = scale(inertia_end, -1, 1.2)
    res = [cognitive_weight, social_weight, inertia_start, inertia_end,
           velocity_max]
    return res


def train_PSO_PSO(function):
    dim = 5
    opso = PSO(dim,
               lambda param: train_mean_PSO(function, minimise, 20, 2000, -5, 10,
                                            *active_PSO(*param)),
               40, 30, inertia_start=0.5, inertia_end=0,
               comparator=minimise, min_bound=MIN_BOUND, max_bound=MAX_BOUND,
               endl='\n\n')
    print('oui')
    opso.run()
    params = active_PSO(*opso.best_position)
    print(params)
    pso = train_PSO(function, minimise, 20, 500, -5, 10, *params)
    print("---", pso.best_score, "---", pso.best_position)
    graph_opso(pso, opso)


if __name__ == '__main__':
    main()
    # ros = Rosenbrock(12)
    # train_PSO_PSO(ros)
