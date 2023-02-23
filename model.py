from random import random
import numpy as np



#Represents the neural network
#Network(2,3,1) would initalize neural network with 2 neurons in input layer, 3 neurons in hidden layer, 1 neuron in output layer
class Network:

    def __init__(self, sizes):
        """Random initalization for biases and weights"""

        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self,a):
        """Returns output for input a to the network"""

        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a
    
    def SGD(self, train_data, epochs, batch_size, eta, test_data = None):
        """Trains neural network with small-batch stochastic gradient descent. train_data is a list of tuples (x,y) representing
        training input and desired output"""

        if test_data:
            n_test = len(test_data)

        n_train = len(train_data)
        for e in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[k : k + batch_size] for k in range(0,n_train,batch_size)]
            for batch in batches:
                self.update_batch(batch,eta)
            if test_data:
                print(f"Epoch {e}: {self.eval(test_data)} / {n_test}")
            else:
                print(f"Epoch {e} complete")
    
    def update_batch(self,batch,eta):
        """Updates weights and biases by applying gradient descent with backpropagation to a single batch"""

        bias_nabla = [np.zeros(b.shape) for b in self.biases]
        weight_nabla = [np.zeros(w.shape) for w in self.weights]

        for x,y in batch:
            delta_bias_nabla, delta_weight_nabla = self.backprop(x,y)
            bias_nabla = [nb + dbn for nb,dbn in zip(bias_nabla,delta_bias_nabla)]
            weight_nabla = [wb + dwn for wb,dwn in zip(weight_nabla,delta_weight_nabla)]

        self.weights = [w - eta/len(batch) * wn for w,wn in zip(self.weights,weight_nabla)]
    
    def backprop(self,x,y):
        """Returns a tuple representing the gradient for cost function"""

        bias_nabla = [np.zeros(b.shape) for b in self.biases]
        weight_nabla = [np.zeros(w.shape) for w in self.weights]

        #feedforward
        activation = x
        activations = [x]
        z_list = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation) + b
            z_list.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #backward pass
        delta  = self.cost_gradient(activations[-1], y) * self.sigmoid_prime(z_list[-1])
        bias_nabla[-1] = delta
        weight_nabla[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.numLayers):
            z = z_list[-1]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(),delta) * sp
            bias_nabla[-1] = delta
            weight_nabla[-1] = np.dot(delta,activations[-l - 1].transpose())
        
        return bias_nabla,weight_nabla
    
    def cost_gradient(self,output_activations,y):
        """Returns vector of partial derivatives for output activations"""

        return output_activations - y

#Out of class functions
def sigmoid(z):
    """Sigmoid function"""

    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    "Derivative of the sigmoid function"

    return sigmoid(z) * (1- sigmoid(z))