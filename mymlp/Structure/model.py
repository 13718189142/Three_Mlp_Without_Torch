import numpy as np
import Structure.activationfun as activationfun
from Structure.Layer import Dense, softmax
import os
import pickle


class ThreeLayerNet:
    '''
        actfunlist: list of activation functions for each layer, e.g. ['relu', 'tanh', 'tanh','leakyrelu'], len =2
        L2: L2 regularization parameter, default=1e-3
    '''

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, L2=1e-3, actfunlist=[]):
        self.L2 = L2
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.layers = []
        self.layers.append(Dense(input_size, hidden_size1, L2=L2))
        self.layers.append(Dense(hidden_size1, hidden_size2, L2=L2))
        self.layers.append(Dense(hidden_size2, output_size, L2=L2))
        self.actfun = []
        self.actfunnames = actfunlist
        valid_activation_functions = ['relu', 'leakyrelu', 'tanh', 'sigmoid']
        if len(actfunlist) != len(self.layers) - 1:
            error_message = f"Number of activation functions provided ({len(actfunlist)}) does not match number of layers ({len(self.layers)-1})."
            raise ValueError(error_message)
        for af in self.actfunnames:
            if af == 'relu':
                self.actfun.append(activationfun.ReLU())
            elif af == 'leakyrelu':
                self.actfun.append(activationfun.LeakyReLu())
            elif af == 'tanh':
                self.actfun.append(activationfun.tangenthyperbolic())
            elif af == 'sigmoid':
                self.actfun.append(activationfun.sigmoid())
            else:
                error_message = f"Invalid activation function '{af}' provided. Please input one of the following values: {', '.join(valid_activation_functions)}."
                raise ValueError(error_message)
        self.reg_loss = 0

    def forward(self, X):
        self.X = X
        self.layers[0].forward(X)
        for i in range(len(self.actfun)):
            self.actfun[i].forward(self.layers[i].output)
            self.layers[i + 1].forward(self.actfun[i].output)
        self.scores = self.layers[len(self.layers) - 1].output
        self.probs = softmax(self.scores)
        return self.probs

    def loss(self, y):
        num_examples = y.shape[0]
        correct_logprobs = -np.log(np.clip(self.probs[range(num_examples), y], 1e-10, 1))
        data_loss = np.sum(correct_logprobs) / num_examples
        self.reg_loss = 0
        for i in range(len(self.layers)):
            self.reg_loss = 0.5 * self.L2 * np.sum(self.layers[i].weights ** 2) + self.reg_loss
        loss = data_loss + self.reg_loss
        return loss

    def backward(self, y):
        num_examples = y.shape[0]
        dscores = self.probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples
        self.layers[len(self.layers) - 1].backward(dscores)
        for i in range(len(self.actfun) - 1, -1, -1):
            self.actfun[i].backward(self.layers[i + 1].dinputs)
            self.layers[i].backward(self.actfun[i].dinputs)

    def update(self, y, learning_rate):
        self.backward(y)
        for i in range(len(self.layers)):
            self.layers[i].weights -= learning_rate * self.layers[i].dweights
            self.layers[i].biases -= learning_rate * self.layers[i].dbiases

    def save(self):
        actfun_str = '_'.join(self.actfunnames)
        struct = f'h1_{self.hidden_size1}h2_{self.hidden_size2}actfun_{actfun_str}L2_{self.L2}'
        model_dir = './Model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_params = {}
        for lay_num in range(len(self.layers)):
            model_params[f'layer_{lay_num}_weights'] = self.layers[lay_num].weights
            model_params[f'layer_{lay_num}_biases'] = self.layers[lay_num].biases

        model_path = os.path.join(model_dir, struct + '_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_params, f)
    def load2(self, folder_path):
        model_path = folder_path
        try:
            with open(model_path, 'rb') as f:
                model_params = pickle.load(f)
            for lay_num in range(len(self.layers)):
                self.layers[lay_num].weights = model_params[f'layer_{lay_num}_weights']
                self.layers[lay_num].biases = model_params[f'layer_{lay_num}_biases']
        except FileNotFoundError:
            print(f"Error: File {model_path} not found. Please check the file path or if the model is saved.")
    def load(self, folder_path):
        actfun_str = '_'.join(self.actfunnames)
        struct = f'h1_{self.hidden_size1}h2_{self.hidden_size2}actfun_{actfun_str}L2_{self.L2}'
        model_path = os.path.join(folder_path, struct + '_model.pkl')
        try:
            with open(model_path, 'rb') as f:
                model_params = pickle.load(f)
            for lay_num in range(len(self.layers)):
                self.layers[lay_num].weights = model_params[f'layer_{lay_num}_weights']
                self.layers[lay_num].biases = model_params[f'layer_{lay_num}_biases']
        except FileNotFoundError:
            print(f"Error: File {model_path} not found. Please check the file path or if the model is saved.")