# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

import random
from copy import deepcopy

INERTIA = 0.7
COGNITIVE_WEIGHT = 1.47
SOCIAL_WEIGHT = 1.47


def minimise(a, b):
    return a <= b


def maximise(a, b):
    return a >= b


class Particle:
    def __init__(self, dimension, position, comparator=maximise):
        self.position = position
        self.dimension = dimension
        self.comparator = comparator
        self.score = 0
        self.default_score = float("inf") if self.comparator(0, 1) \
            else -float("inf")
        self.velocity = [0 for _ in range(dimension)]
        self.informants = []
        self.best_score = self.default_score
        self.best_position = self.position

    def evaluate(self, fitness_function):
        self.score = fitness_function(self.position)

    def get_best_informant_position(self):
        best_score = self.default_score
        best_position = []
        for particle in self.informants:
            if self.comparator(particle.best_score, best_score):
                best_score = particle.best_score
                best_position = deepcopy(particle.best_position)
        return best_position

    def update_velocity(self):
        best_informant_position = self.get_best_informant_position()
        for i in range(self.dimension):
            self.velocity[i] = (
                INERTIA * self.velocity[i]
                + random.uniform(0, 1) * COGNITIVE_WEIGHT
                * (self.best_position[i] - self.position[i])
                + random.uniform(0, 1) * SOCIAL_WEIGHT
                * (best_informant_position[i] - self.position[i])
            )

    def update_best_position(self):
        if self.comparator(self.score, self.best_score):
            self.best_score = self.score
            self.best_position = deepcopy(self.position)

    def move(self, min_bound, max_bound):
        for i in range(self.dimension):
            self.position[i] = max(min_bound,
                                   min(max_bound,
                                       self.position[i] + self.velocity[i]))


class PSO:
    def __init__(self, dimension, fitness_function, max_iter,
                 comparator=maximise, min_bound=-10, max_bound=10):
        if dimension <= 0:
            raise ValueError('The vector dimension should be greater than 0')
        self.dimension = dimension
        self.fitness_function = fitness_function
        if max_iter <= 0:
            raise ValueError('The max iteration should be greater than 0')
        self.max_iter = max_iter
        self.comparator = comparator
        self.max_bound = max_bound
        self.min_bound = min_bound
        particle_nb = 40
        self.particles = [
            Particle(dimension, self.generate_position(), comparator)
            for _ in range(particle_nb)
        ]
        for particle in self.particles:
            particle.evaluate(fitness_function)
            idx = self.particles.index(particle)
            particle.informants.append(self.particles[idx - 1])
            particle.informants.append(
               self.particles[(idx + 1) % particle_nb]
            )
            # particle.informants = [i for i in self.particles if i != particle]

    def generate_position(self):
        return [random.uniform(self.min_bound, self.max_bound)
                for _ in range(self.dimension)]

    def run(self):
        best_score = float("inf") if self.comparator(0, 1) else -float("inf")
        best_position = []
        for i in range(self.max_iter):
            print('%d / %d' % (i, self.max_iter), end="\r")
            for particle in self.particles:
                if self.comparator(particle.score, best_score):
                    best_score = particle.score
                    best_position = deepcopy(particle.position)
            for particle in self.particles:
                particle.update_velocity()
                particle.move(self.min_bound, self.max_bound)
                particle.evaluate(self.fitness_function)
                particle.update_best_position()
        return best_score, best_position
