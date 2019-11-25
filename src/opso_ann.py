# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset:4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble


from pso import PSO, minimise
from pso_ann import train_ANN_PSO
import train_help
from pso_json import get_boundary_config, decode_args, encode_args
import matplotlib.pyplot as plt
from args import opso_args


def scale_args(args, boundary):
    # Iterate through all arguments to scale them between specific born
    i = 0
    for key in boundary:
        args[i] = train_help.scale(args[i], boundary[key][0], boundary[key][1])
        i += 1

    # Round nb_h_layers and nb_neurons_layer to have int values
    args[1] = round(args[1])
    args[2] = round(args[2])

    # Get activation functions
    i_activation = round(train_help.scale(args[-1], 0,
                                          len(train_help.ACTIVATIONS) - 1))
    activations = [train_help.ACTIVATIONS[i_activation]
                   for _ in range(args[1] + 1)]
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


def train_PSO_PSO_ANN(inputs, res_ex, boundary, opso_arg, pso_arg,
                      draw_graph=False):
    dim = 11
    opso = PSO(dim,
               lambda param: fitness_mean(inputs, res_ex, *pso_arg.values(),
                                          *scale_args(param, boundary)),
               **opso_arg, comparator=minimise,
               min_bound=train_help.MIN_BOUND, max_bound=train_help.MAX_BOUND,
               endl="11")
    print("\nRunning...\n")
    if draw_graph:
        opso.set_graph_config(inputs=inputs, res_ex=res_ex, opso=True)
    opso.run()
    return opso


def main():
    args = opso_args().parse_args()
    file_name = train_help.name_to_file(args.function)
    inputs, res_ex = train_help.read_input(file_name)
    opso_arg = decode_args('', 'opso', args.onc)
    real_time_graph = False
    boundary = get_boundary_config(args.obc)
    pso = train_PSO_PSO_ANN(inputs, res_ex, boundary, **opso_arg,
                            draw_graph=real_time_graph)
    dict_pso = {**train_help.args_to_pso_kwargs(
        scale_args(pso.best_position, boundary)),
                **opso_arg["pso_arg"]}
    train_help.write_activation(dict_pso)
    encode_args(args.function, 'pso', **dict_pso)
    if not real_time_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex, opso=True)
        pso.draw_graphs()
    plt.show()


if __name__ == '__main__':
    main()
