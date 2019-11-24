# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

import random
from copy import deepcopy
import matplotlib.pyplot as plt

INERTIA_START = 0.9
INERTIA_END = 0.4
COGNITIVE_WEIGHT = 2
SOCIAL_WEIGHT = 2
VELOCITY_MAX = 20


def minimise(a, b):
    return a <= b


def maximise(a, b):
    return a >= b


class Particle:
    def __init__(self, dimension, position, min_bound, max_bound,
                 comparator=maximise,
                 cognitive_weight=COGNITIVE_WEIGHT,
                 social_weight=SOCIAL_WEIGHT, inertia_start=INERTIA_START,
                 inertia_end=INERTIA_END, velocity_max=VELOCITY_MAX):
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia_start = inertia_start
        self.inertia_end = inertia_end
        self.velocity_max = velocity_max
        self.position = position
        self.dimension = dimension
        self.comparator = comparator
        self.score = 0
        self.res = dimension
        self.default_score = float("inf") if self.comparator(0, 1) \
            else -float("inf")
        self.velocity = [random.uniform(min_bound-position[i],
                                        max_bound - position[i])
                         for i in range(dimension)]
        self.neighbors = []
        self.best_score = self.default_score
        self.best_position = self.position

    def evaluate(self, fitness_function):
        self.score, self.res = fitness_function(self.position)

    def get_best_informant_position(self):
        best_score = self.default_score
        best_position = []
        for particle in self.neighbors:
            if self.comparator(particle.best_score, best_score):
                best_score = particle.best_score
                best_position = deepcopy(particle.best_position)
        return best_position

    def update_velocity_2011(self, inertia):
        best_informant_position = self.get_best_informant_position()
        for i in range(self.dimension):
            gravity = (self.position[i]
                       + (random.uniform(0, 1) * self.cognitive_weight)
                       * (self.best_position[i] - self.position[i])
                       + (random.uniform(0, 1) * self.social_weight)
                       * (best_informant_position[i] - self.position[i]))
            random_new_position = random.uniform(gravity, self.position[i])
            self.velocity[i] = min(self.velocity_max,
                                   inertia * self.velocity[i]
                                   + random_new_position - self.position[i])

    def update_velocity(self, inertia):
        best_informant_position = self.get_best_informant_position()
        for i in range(self.dimension):
            self.velocity[i] = min(
                self.velocity_max,
                (inertia * self.velocity[i]
                 + random.uniform(0, 1) * self.cognitive_weight
                 * (self.best_position[i] - self.position[i])
                 + random.uniform(0, 1) * self.social_weight
                 * (best_informant_position[i] - self.position[i]))
            )

    def update_best_position(self):
        if self.comparator(self.score, self.best_score):
            self.best_score = self.score
            self.best_position = deepcopy(self.position)

    def move(self):
        for i in range(self.dimension):
            self.position[i] = max(self.min_bound,
                                   min(self.max_bound,
                                       self.position[i] + self.velocity[i]))


class PSO:
    def __init__(self, dimension, fitness_function, max_iter=100, n_particle=40,
                 n_neighbor=4, cognitive_weight=COGNITIVE_WEIGHT,
                 social_weight=SOCIAL_WEIGHT, inertia_start=INERTIA_START,
                 inertia_end=INERTIA_END, velocity_max=VELOCITY_MAX,
                 comparator=maximise, min_bound=-10, max_bound=10, endl='\r',
                 version=2007):
        self.version = version
        if dimension <= 0:
            raise ValueError('The vector dimension should be greater than 0')
        self.dimension = dimension
        self.fitness_function = fitness_function
        if max_iter <= 0:
            raise ValueError('The max iteration should be greater than 0')
        self.max_iter = max_iter
        self.comparator = comparator
        self.inertia_start = inertia_start
        self.inertia_end = inertia_end
        self.max_bound = max_bound
        self.min_bound = min_bound
        self.best_score = float("inf") if self.comparator(0, 1) \
            else -float("inf")
        self.best_position = []
        self.average_mean_square_error = []
        self.best_mean_square_error = []
        self.graph_config = {}
        self.best_res = []
        self.endl = endl
        self.particles = [
            Particle(dimension, self.generate_position(), min_bound, max_bound,
                     comparator,
                     cognitive_weight, social_weight, velocity_max)
            for _ in range(n_particle)
        ]
        if n_neighbor < 0:
            raise ValueError('The nb of informant should be greater than 0')
        if n_neighbor > n_particle - 1:
            raise ValueError('The nb of informant should be smaller than the '
                             'number of particle - 1')
        for particle in self.particles:
            particle.evaluate(fitness_function)
            idx = self.particles.index(particle)
            for i in range(1, int(n_neighbor / 2) + 1):
                particle.neighbors.append(self.particles[idx - i])
                particle.neighbors.append(
                   self.particles[(idx + i) % n_particle]
                )
            # particle.neighbors = [i for i in self.particles if i != particle]

    def generate_position(self):
        return [random.uniform(self.min_bound, self.max_bound)
                for _ in range(self.dimension)]

    def run(self):
        for i in range(self.max_iter):
            print('%d / %d' % (i+1, self.max_iter), end=self.endl)
            inertia = self.inertia_start \
                - ((self.inertia_start - self.inertia_end) / self.max_iter) * i
            best_local_score = self.particles[0].score
            for particle in self.particles:
                if self.comparator(particle.score, self.best_score):
                    self.best_score = particle.score
                    self.best_position = deepcopy(particle.position)
                    self.best_res = particle.res
                if self.comparator(particle.score, best_local_score):
                    best_local_score = particle.score
            self.best_mean_square_error.append(best_local_score)
            self.average_mean_square_error.append(
                sum(x.score for x in self.particles) / len(self.particles))
            for particle in self.particles:
                if self.version == 2007:
                    particle.update_velocity(inertia)
                elif self.version == 2011:
                    particle.update_velocity_2011(inertia)
                else:
                    raise ValueError('Wrong PSO Version')
                particle.move()
                particle.evaluate(self.fitness_function)
                particle.update_best_position()
            if self.graph_config:
                self.draw_graphs()

    def set_graph_config(self, res_ex, inputs, dry):
        inputs_str = [f"{i}: {inputs[i]}" for i in range(len(inputs))]
        self.graph_config = {
            'res_ex': res_ex,
            'inputs': inputs_str,
            'dry': dry
        }
        plt.figure(1)
        self.graph_config['ann_ax'] = plt.subplot(212)
        self.graph_config['pso_ax'] = plt.subplot(211 if dry else 221)
        if not dry:
            self.graph_config['opso_ax'] = plt.subplot(222)

    @staticmethod
    def draw_graph_pso(pso, ax, name="PSO"):
        ax.clear()
        plt.subplot(ax)
        plt.title(name + " Mean square error evolution")
        plt.plot(pso.best_mean_square_error, color='g', label='Best')
        plt.plot(pso.average_mean_square_error, color='c', label='Average')
        plt.legend()

    def draw_graph_ann(self, res):
        self.graph_config['ann_ax'].clear()
        plt.subplot(self.graph_config['ann_ax'])
        plt.title("Target output and the ANN output comparaison")
        plt.plot(self.graph_config['inputs'], res, label='Result')
        plt.plot(self.graph_config['inputs'], self.graph_config['res_ex'],
                 linestyle=':', label='Target')
        plt.legend()
        plt.tick_params(axis='x', labelrotation=70, width=0.5)
        plt.xticks(range(0, len(self.graph_config['inputs']), 5))

    def draw_graphs(self):
        if self.graph_config['dry']:
            self.draw_graph_ann(self.best_res)
            self.draw_graph_pso(self, self.graph_config['pso_ax'])
        else:
            self.draw_graph_ann(self.best_res.best_res)
            self.draw_graph_pso(self.best_res, self.graph_config['pso_ax'])
            self.draw_graph_pso(self, self.graph_config['opso_ax'], "OPSO")
        plt.pause(0.0005)


class Rosenbrock:
    def __init__(self, dimension=2):
        self.max_bound = 10
        self.min_bound = -5
        self.dimension = dimension

    def generate_random(self):
        return [random.uniform(self.min_bound, self.max_bound)
                for _ in range(self.dimension)]

    def evaluate(self, xx):
        d = len(xx)
        int_sum = 0
        for i in range(d-1):
            xi = xx[i]
            xnext = xx[i+1]
            new = 100*(xnext-xi**2)**2 + (xi-1)**2
            int_sum = int_sum + new
        y = int_sum
        return y
