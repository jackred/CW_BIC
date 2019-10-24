# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>,
# Timothée Couble

from ann_help import default_activation, default_weights


class Connection:
    def __init__(self, layer1, layer2, weight=default_weights):
        self.layer1 = layer1
        self.layer2 = layer2
        self.weights = [[weight() for j in range(layer2.length)]
                        for i in range(layer1.length)]

    def compute_layers(self):
        for i in range(self.layer2.length):
            tmp = 0
            for j in range(self.layer1.length):
                tmp += (self.layer1.neurons[j]
                        * self.weights[j][i]
                        + self.layer2.bias)
            self.layer2.neurons[i] = self.layer2.activation(tmp)


class Layer:
    def __init__(self, nb_neurons, bias=0, activation=default_activation):
        self.neurons = [0] * nb_neurons
        self.bias = bias
        self.length = nb_neurons
        self.activation = activation


class Input_Layer(Layer):
    def __init__(self, nb_neurons):
        self.neurons = [0] * nb_neurons
        self.length = nb_neurons


class ANN:
    def __init__(self, nb_neurons, nb_layers, bias=[],
                 activations=[], weight=default_weights):
        bias = bias if len(bias) == nb_layers-1 else [0] * (nb_layers-1)
        activations = activations if len(activations) == nb_layers-1 \
            else [default_activation] * (nb_layers-1)
        if nb_layers < 0:
            raise ValueError('There should be at least one layer')
        if [i for i in nb_neurons if i <= 0] != []:
            raise ValueError('Layers should have at least one neuron')
        if len(nb_neurons) != nb_layers:
            raise ValueError(
                'Numbers of neurons should match numbers of layers')
        self.layers = [Layer(nb_neurons[i], bias[i-1], activations[i-1])
                       for i in range(1, nb_layers)]
        self.layers.insert(0, Input_Layer(nb_neurons[0]))
        self.connections = [Connection(self.layers[i],
                                       self.layers[i+1],
                                       weight)
                            for i in range(nb_layers - 1)]

    def activation(self, input_nodes):
        self.connections[0].layer1.neurons = input_nodes
        for connection in self.connections:
            connection.compute_layers()
        return self.layers[-1].neurons
