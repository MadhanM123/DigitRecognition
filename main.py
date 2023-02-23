import mnist_loader
import network

def main():
    training_data,val_data,test_data = mnist_loader.load_data_wrapper()
    alg = network.Network([784,30,10])
    alg.SGD(training_data,30,10,3.0,test_data=test_data)

if(__name__ == "__main__"):
    main()