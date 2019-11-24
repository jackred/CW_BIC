# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

from pso import PSO, maximise, minimise
import train_help
from math import sqrt


def train_PSO(function, comparator,  n_iter, n_particle, n_neighbor, min_bound,
              max_bound,
              cognitive_trust, social_trust, inertia_start, inertia_end,
              velocity_max):
    pso = PSO(function.dimension, function.evaluate, max_iter=n_iter,
              n_particle=n_particle, n_neighbor=n_neighbor,
              comparator=comparator,
              min_bound=min_bound, max_bound=max_bound,
              cognitive_trust=cognitive_trust, social_trust=social_trust,
              inertia_start=inertia_start, inertia_end=inertia_end,
              velocity_max=velocity_max, version=2011, endl='\r')
    pso.run()
    return pso


def main():
    rosenbrock = train_help.Rosenbrock(12)
    res = []
    for i in range(30):
        pso = train_PSO(rosenbrock, minimise, 2000, 20, 2, -5, 10,
                        2, 2, 0.8, 0.4, 20)
        res.append(pso.best_global_score)
    res.sort()
    print(res)
    mean = sum(res) / len(res)
    print('mean:', mean)
    print('median:', res[len(res)//2])
    print('std:', sqrt((1/(len(res) - 1)) * sum([(i - mean) ** 2
                                                 for i in res])))
    print('best:', res[0])
    print('worst:', res[-1])


if __name__ == '__main__':
    main()
