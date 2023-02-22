from tensorflow import keras
from keras.datasets import mnist
import numpy as np


#Represents the neural network
#Network(2,3,1) would initalize neural network with 2 neurons in input layer, 3 neurons in hidden layer, 1 neuron in output layer
class Network:

    #Random initalization for biases and weights
    def __init__(self, sizes):
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]