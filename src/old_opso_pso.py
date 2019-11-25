# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset:4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble


from pso import PSO, minimise
from test_pso import train_PSO
from train_help import graph_opso, read_input
from ann_help import ACTIVATIONS, scale, MIN_BOUND, MAX_BOUND, Rosenbrock


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
               lambda param: train_mean_PSO(function, minimise, 20, 2000, -5,
                                            10, *active_PSO(*param)),
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


def main():
    ros = Rosenbrock(12)
    train_PSO_PSO(ros)


if __name__ == '__main__':
    main()
