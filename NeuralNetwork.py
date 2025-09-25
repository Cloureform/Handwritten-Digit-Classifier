import numpy as np
class neuralNework:
    def __init__(self, input_size = 784, hidden_layers = [256, 256], output_size = 10):
        self.input_size = input_size
        self.hidden_layer = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.bias = []

        #connecting input -> hidden layer
        #Adding the random weights
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.bias.append(np.zeros((1, hidden_layers[0]))) #2d array of value zero

        #connecting hidden layer -> hidden layer
        for i in range(len(hidden_layers) - 1): #This way I can change the number of hidden layers if I want
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i+1]))
            self.bias.append(np.zeros((1, hidden_layers[i+1])))

        #connecting hidden layer -> output
        self.weights.append(0.01 * np.random.randn(hidden_layers[len(hidden_layers)-1], output_size))
        self.bias.append(np.zeros((1, output_size)))
        
    def forward_propagation(self, inputs):
        layers = [inputs]
        #Linear transformation
        for i in range(len(self.weights)):# 1, 2, 3, 4, 5,6 
            z = np.dot(layers[-1], self.weights[i] + self.bias[i])
            layers.append(z) #just in case
            
        #Activation function()
            if i == len(self.weights) - 1: #softmax
                output = np.exp(z - np.max(z, axis=1, keepdims=True))
                output = output / np.sum(output, axis=1, keepdims=True)
            else: #ReLU
                output = z[z<0] =0 #np.maximum(z, 0) slower than the index method on the left

            layers.append(output)



