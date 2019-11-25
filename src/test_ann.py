# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed  this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from ann import ANN
from pso import PSO, maximise, minimise, plt
import train_help


def fitness_for_ann(params, ann, inputs, res_ex):
    ann.update_params(params)
    res = []
    for i in range(len(inputs)):
        estimation = ann.activation(inputs[i])
        res.append(estimation[0])
    return train_help.mean_square_error(res_ex, res), res


def train_ANN_PSO(inputs, res_ex, n_iter, n_particle, n_neighbor, nb_h_layers,
                  nb_neurons_layer, min_bound, max_bound, cognitive_trust,
                  social_trust, inertia_start, inertia_end,
                  velocity_max, activation, draw_graph=False):
    nb_neurons = [len(inputs[0])]
    if type(nb_neurons_layer) == list:
        nb_neurons.extend(nb_neurons_layer)
    else:
        nb_neurons.extend([nb_neurons_layer] * nb_h_layers)
    nb_neurons.append(1)
    print(nb_neurons, n_neighbor, activation)
    ann = ANN(nb_neurons=nb_neurons, nb_layers=len(nb_neurons),
              activations=activation)
    dim = sum(nb_neurons[i] * nb_neurons[i+1]
              for i in range(len(nb_neurons)-1)) + len(nb_neurons) - 1
    pso = PSO(dim, lambda params: fitness_for_ann(params, ann, inputs, res_ex),
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
    inputs, res_ex = train_help.read_input(name)
    real_time_graph = False
    # args = [2, 1, 3, -7.176582343826539, 3.0666915574121836, 0.0,
    #         2.262511221377284, -0.2638196189084406, 1.0, 50.0, train_help.atan]
    args = [2, 2, 5, -3.0, 0.0, 8.0, 3.30281688745529, 2.0, -1.214638225596609,
            38.52018177792397, [train_help.sigmoid, train_help.sigmoid,
                                train_help.sigmoid]]
    pso, ann = train_ANN_PSO(inputs, res_ex, 300, 60, *args,
                             draw_graph=real_time_graph)
    if not real_time_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex)
        pso.draw_graphs()
    plt.show()


if __name__ == '__main__':
    main()
