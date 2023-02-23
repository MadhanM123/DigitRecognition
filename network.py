from random import shuffle
import numpy as np

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

        train_data = list(train_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        n_train = len(train_data)
        for e in range(epochs):
            shuffle(train_data)
            batches = [train_data[k : k + batch_size] for k in range(0,n_train,batch_size)]
            for batch in batches:
                self.update_batch(batch,eta)
            if test_data:
                print(f"Epoch {e}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {e} complete")
    
    def update_batch(self,batch,eta):
        """Updates weights and biases by applying gradient descent with backpropagation to a single batch"""

        bias_nabla = [np.zeros(b.shape) for b in self.biases]
        weight_nabla = [np.zeros(w.shape) for w in self.weights]

        for x,y in batch:
            delta_bias_nabla, delta_weight_nabla = self.backprop(x,y)
            weight_nabla = [wb + dwn for wb,dwn in zip(weight_nabla,delta_weight_nabla)]
            bias_nabla = [nb + dbn for nb,dbn in zip(bias_nabla,delta_bias_nabla)]
            
        self.weights = [w - eta/len(batch) * wn for w,wn in zip(self.weights,weight_nabla)]
        self.biases = [b - eta/len(batch) * bn for b,bn in zip(self.biases,bias_nabla)]

        
    
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
        delta  = self.cost_gradient(activations[-1], y) * sigmoid_prime(z_list[-1])
        bias_nabla[-1] = delta
        weight_nabla[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.numLayers):
            z = z_list[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(),delta) * sp
            bias_nabla[-l] = delta
            weight_nabla[-l] = np.dot(delta,activations[-l - 1].transpose())
        
        return bias_nabla,weight_nabla
    
    def cost_gradient(self,output_activations,y):
        """Returns vector of partial derivatives for output activations"""

        return output_activations - y
    
    def evaluate(self, test_data):
        "Evaluates accuracy for test data"

        results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in results)


#Out of class functions
def sigmoid(z):
    """Sigmoid function"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    "Derivative of the sigmoid function"
    return sigmoid(z) * (1- sigmoid(z))



