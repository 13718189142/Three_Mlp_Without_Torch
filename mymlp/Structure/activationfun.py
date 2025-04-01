import numpy as np
class ReLU:
    def forward(self, inputs):
        self.mem_input = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.mem_input <= 0] = 0                   
        return self.dinputs

class LeakyReLu:
    def forward(self, inputs,leacky=0.1):
        self.mem_input = inputs
        self.leacky_param=leacky
        self.output = np.maximum(self.leacky_param*inputs,inputs)
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.mem_input <= 0] = self.leacky_param*self.dinputs[self.mem_input <= 0]   
        return self.dinputs


class tangenthyperbolic:
    def forward(self, inputs):
        self.mem_input = inputs
        self.output = np.tanh(self.mem_input)
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues*(1.0 - np.power((self.output),2)) # sech^2{x}  
        return self.dinputs

class sigmoid:
    def forward(self, inputs):
        self.output = 1/ (1+ np.exp(-inputs))
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
        return self.dinputs 


# class Softmax:
#     def forward(self, inputs):
#         self.inputs     = inputs
#         exp_values      = np.exp(inputs -   np.max(inputs       , axis=1,keepdims=True))
#         probabilities   = exp_values /      np.sum(exp_values   , axis=1,keepdims=True)
#         self.output = probabilities
#         return self.output  
#     def backward(self, dvalues):
#         self.dinputs = np.empty_like(dvalues)
#         for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
#             single_output = single_output.reshape(-1, 1)
#             jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
#             self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
