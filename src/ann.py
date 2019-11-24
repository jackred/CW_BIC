# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>,
# Timothée Couble

from ann_help import default_activation

class Connection:
    """
    represent a connection between 2 layers:
    layer1 and layer2
    weights is a list of weights between each neurons of the 2 layer
    """
    def __init__(self, layer1, layer2,):
        self.layer1 = layer1
        self.layer2 = layer2
        self.weights = []

    """
    transfer function
    compute the value of layer2 with the value from layer1
    """
    def compute_layers(self):
        for i in range(self.layer2.length):
            tmp = 0
            for j in range(self.layer1.length):
                tmp += (self.layer1.neurons[j]  # activation of layer1
                        * self.weights[j][i]  # multiply by the weights
                        + self.layer2.bias)  # add the bias of layer2
            # activation of layer2
            self.layer2.neurons[i] = self.layer2.activation(tmp)


class Layer:
    """
    represent a layer of neurons
    contain N neurons
    is associated to a bias and an activation function
    """
    def __init__(self, nb_neurons, activation):
        self.neurons = [0] * nb_neurons
        self.length = nb_neurons
        self.bias = 0
        self.activation = activation


class InputLayer(Layer):
    """
    represent the input layer
    contain N neurons
    """
    def __init__(self, nb_neurons):
        self.neurons = [0] * nb_neurons
        self.length = nb_neurons


class ANN:
    """
    Artificial neural network
    Contain [N0, N1, ..., Nn-1] neurons, with n layers
    """
    def __init__(self, nb_neurons, nb_layers, activations):
        self.nb_weights = sum(nb_neurons[i] * nb_neurons[i+1]
                              for i in range(len(nb_neurons)-1))
        if nb_layers < 0:
            raise ValueError('There should be at least one layer')
        if [i for i in nb_neurons if i <= 0] != []:
            raise ValueError('Layers should have at least one neuron')
        if len(nb_neurons) != nb_layers:
            raise ValueError(
                'Numbers of neurons should match numbers of layers')
        self.layers = [Layer(nb_neurons[i], activations[i-1])
                       for i in range(1, nb_layers)]
        self.layers.insert(0, InputLayer(nb_neurons[0]))
        self.connections = [Connection(self.layers[i],
                                       self.layers[i+1])
                            for i in range(nb_layers - 1)]
    """
    evaluate the given inputs throught the neural network
    compute the value and return a result
    """
    def activation(self, input_nodes):
        self.connections[0].layer1.neurons = input_nodes
        for connection in self.connections:
            connection.compute_layers()
        return self.layers[-1].neurons

    """
    replace the weights (training)
    """
    def update_weights(self, weights):
        weights = self.format_weights(weights)
        for i in range(len(self.connections)):
            self.connections[i].weights = weights[i]

    """
    replace the bias (training)
    """
    def update_bias(self, bias):
        for i in range(len(self.layers)-1):
            self.layers[i+1].bias = bias[i]

    """
    format the returned params and update them (training)
    """
    def update_params(self, params):
        m = 0
        weights = params[m:self.nb_weights]
        m += self.nb_weights
        bias = params[m:m+len(self.layers)-1]  # -1: no bias on input layer
        self.update_weights(weights)
        self.update_bias(bias)

    """
    format the weights according to the number of neurons per layer
    """
    def format_weights(self, weights):
        res = []
        j = 0
        for i in range(len(self.layers) - 1):
            tmp = []
            for k in range(len(self.layers[i].neurons)):
                nb_neurons_i = len(self.layers[i+1].neurons)
                tmp.append(weights[j:j+nb_neurons_i])
                j += nb_neurons_i
            res.append(tmp)
        return res
