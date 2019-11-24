# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset:4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble


from pso import PSO, minimise
from test_ann import train_ANN_PSO
from test_pso import train_PSO
from train_help import graph_opso, read_input
from pso_json import get_born_config
from ann_help import ACTIVATIONS, scale, MIN_BOUND, MAX_BOUND, Rosenbrock
import matplotlib.pyplot as plt


def active(args, born):
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


def train_mean(*args):
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


def train_PSO_PSO_ANN(inputs, res_ex, borns, draw_graph=False):
    dim = 11
    opso = PSO(dim,
               lambda param:
               train_mean(inputs, res_ex, 50, 25, *active(param, borns)),
               max_iter=2, n_particle=7, n_neighbor=4,
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
    print(active(pso.best_position, born))
    if not real_time_graph:
        pso.set_graph_config(inputs=inputs, res_ex=res_ex, opso=True)
        pso.draw_graphs()
    plt.show()


def train_mean_PSO(*args):
    res = []
    for i in range(4):
        pso = train_PSO(*args)
        res.append(pso.best_global_score)
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
               40, 30, inertia_start=0.5, inertia_end=0.5,
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
