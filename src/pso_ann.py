# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed  this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from ann import ANN
from args import pso_args
from pso import PSO, maximise, minimise, plt
import train_help
from pso_json import decode_args, encode_args
from copy import deepcopy


def fitness_for_ann(params, ann, inputs, res_ex):
    ann.update_params(params)
    res = []
    for i in range(len(inputs)):
        estimation = ann.activation(inputs[i])
        res.append(estimation[0])
    return train_help.mean_square_error(res_ex, res), res


def set_nb_neurons(n_input, nb_neurons_layer, nb_h_layers):
    nb_neurons = [n_input]
    if type(nb_neurons_layer) == list:
        nb_neurons.extend(nb_neurons_layer)
    else:
        nb_neurons.extend([nb_neurons_layer] * nb_h_layers)
    nb_neurons.append(1)
    return nb_neurons


def train_ANN_PSO(inputs, res_ex, max_iter, n_particle, n_neighbor,
                  nb_h_layers,
                  nb_neurons_layer, min_bound, max_bound, cognitive_trust,
                  social_trust, inertia_start, inertia_end,
                  velocity_max, activations, draw_graph=False):
    nb_neurons = set_nb_neurons(len(inputs[0]), nb_neurons_layer, nb_h_layers)
    # print(nb_neurons, n_neighbor, activations)
    ann = ANN(nb_neurons=nb_neurons, nb_layers=len(nb_neurons),
              activations=activations)
    dim = sum(nb_neurons[i] * nb_neurons[i+1]
              for i in range(len(nb_neurons)-1)) + len(nb_neurons) - 1
    pso = PSO(dim, lambda params: fitness_for_ann(params, ann, inputs, res_ex),
              max_iter=max_iter, n_particle=n_particle,  n_neighbor=n_neighbor,
              comparator=minimise,
              min_bound=min_bound, max_bound=max_bound,
              cognitive_trust=cognitive_trust, social_trust=social_trust,
              inertia_start=inertia_start, inertia_end=inertia_end,
              velocity_max=velocity_max, endl='', version=2011)
    if draw_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex)
    pso.run()
    return pso, ann


def main():
    args = pso_args().parse_args()
    file_name = train_help.name_to_file(args.function)
    inputs, res_ex = train_help.read_input(file_name)
    real_time_graph = False
    pso_arg = decode_args(args.function, 'pso', args.pnc)
    activations = deepcopy(pso_arg['activations'])
    train_help.read_activation(pso_arg)
    pso, ann = train_ANN_PSO(inputs, res_ex, **pso_arg,
                             draw_graph=real_time_graph)
    print(pso_arg)
    nb_neurons = set_nb_neurons(len(inputs[0]), pso_arg['nb_neurons_layer'],
                                pso_arg['nb_h_layers'])
    encode_args(args.function, 'ann', params=pso.best_position,
                nb_neurons=nb_neurons, nb_layers=len(nb_neurons),
                activations=activations)
    if not real_time_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex)
        pso.draw_graphs()
    plt.show()


if __name__ == '__main__':
    main()
