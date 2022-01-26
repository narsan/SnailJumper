import numpy as np

from ctypes.wintypes import PFILETIME
from typing import Counter


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # layer_sizes example: [3, 10, 2]

        self.input_layer_size, self.hidden_layer_size, self.output_layer_size = layer_sizes

        # WEIGHTS
        self.W1 = np.random.normal(size=(self.hidden_layer_size, self.input_layer_size))
        self.W2 = np.random.normal(size=(self.output_layer_size, self.hidden_layer_size))

        # BIASES
        self.b1 = np.zeros((self.hidden_layer_size, 1))
        self.b2 = np.zeros((self.output_layer_size, 1))

    def activation(self, x):
        # sigmoid function
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        x = x.reshape((self.input_layer_size, 1))
        a1 = self.activation(self.W1 @ x + self.b1)
        a2 = self.activation(self.W2 @ a1 + self.b2)

        return a2
