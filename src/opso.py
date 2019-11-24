# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset:4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from pso import PSO, minimise
from test_ann import train_ANN_PSO, read_input
from ann_help import ACTIVATIONS, scale, MIN_BOUND, MAX_BOUND
import matplotlib.pyplot as plt

def active(nb_h_layers, nb_neurons_layer,
           min_bound, max_bound, cognitive_weight,
           social_weight, inertia_start, inertia_end,
           velocity_max, i_activation):
    i_activation = round(scale(i_activation, 0, len(ACTIVATIONS) - 1))
    velocity_max = scale(velocity_max, 0.000001, 50)
    social_weight = scale(social_weight, 0, 4)
    cognitive_weight = scale(cognitive_weight, 0, 4)
    inertia_start = scale(inertia_start, -1, 1)
    inertia_end = scale(inertia_end, -1, 1)
    min_bound = scale(min_bound, -10, 0)
    max_bound = scale(max_bound, 0.00001, 10)
    nb_h_layers = round(scale(nb_h_layers, 1, 6))
    nb_neurons_layer = round(scale(nb_neurons_layer, 1, 4))
    res = [nb_h_layers, nb_neurons_layer,
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


def train_PSO_PSO(inputs, res_ex, draw_graph=False):
    dim = 10
    opso = PSO(dim,
               lambda param: train_mean(inputs, res_ex, 5, 20,
                                        *active(*param)),
               10, 3, inertia_start=0.5, inertia_end=0.5,
               comparator=minimise, min_bound=MIN_BOUND, max_bound=MAX_BOUND)
    print("\nRunning...\n")
    if draw_graph:
        opso.set_graph_config(inputs=inputs, res_ex=res_ex, dry=False)
    opso.run()
    return opso


def main():
    name = '../Data/1in_tanh.txt'
    inputs, res_ex = read_input(name)
    real_time_graph = False
    pso = train_PSO_PSO(inputs, res_ex, draw_graph=real_time_graph)
    if not real_time_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex, dry=False)
        pso.draw_graphs()
    plt.show()


if __name__ == '__main__':
    main()