# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset:4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from pso import PSO, minimise
from test_ann import train_ANN_PSO, read_input, graph
from ann_help import ACTIVATIONS, scale, MIN_BOUND, MAX_BOUND


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


def train_PSO_PSO(name):
    inputs, res_ex = read_input(name)
    dim = 10
    opso = PSO(dim,
               lambda param: train_ANN_PSO(inputs, res_ex, 20, 150,
                                           *active(*param))[0].best_score,
               20, 20, inertia_start=0.5, inertia_end=0.5,
               comparator=minimise, min_bound=MIN_BOUND, max_bound=MAX_BOUND)
    print('oui')
    opso.run()
    params = active(*opso.best_position)
    print(params)
    pso, ann = train_ANN_PSO(inputs, res_ex, 40, 300, *params)
    graph(pso, ann, res_ex, inputs)


if __name__ == '__main__':
    name = '../Data/1in_linear.txt'
    train_PSO_PSO(name)
