import  numpy as np
class Dense:
    def __init__(self, n_inputs, n_neurons, mean=0, variance=1, bias=0, L2=0):
        self.weights = np.random.normal(mean, (variance / n_neurons), [n_inputs, n_neurons]) + bias
        self.biases = np.random.uniform(-1, 1, size=(1, n_neurons)) * variance
        self.L2 = L2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.L2 > 0:
            self.dweights += 2 * self.L2 * self.weights
            self.dbiases += 2 * self.L2 * self.biases
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)