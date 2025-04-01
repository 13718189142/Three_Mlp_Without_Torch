import numpy as np
from Structure.model import ThreeLayerNet
from train import train
from test import test
from parameter_search import parameter_search
from dataloaders import load_cifar10
import pickle
import  os
import os




if __name__ == "__main__":
    #cifar10_dir = r'./cifar-10-batches-py'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cifar10_dir = os.path.join(current_dir, 'cifar-10-batches-py')
    model_dir = './mymlp/Model'

    X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    num_training = 49000
    num_validation = 1000
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    input_size = X_train.shape[1]
    output_size = 10


    model = ThreeLayerNet(input_size, 512,256, output_size,L2=0,actfunlist=['relu','leakyrelu'])
    #model.forward(X_train)
    model,train_losses, val_losses, val_accuracies = train(model,X_train, y_train, X_val, y_val,learning_rate=1e-4,num_epochs=1)
    test_acc = test(model, X_test, y_test)
    print('Test accuracy: %f' % test_acc)
    model.save()

    test_model = ThreeLayerNet(input_size, 512,256, output_size,L2=0,actfunlist=['relu','leakyrelu'])
    test_model.load(model_dir)
    test_acc = test(test_model, X_test, y_test)
    print('Test accuracy: %f' % test_acc)
    #best_model = parameter_search(X_train, y_train, X_val, y_val, X_test, y_test,actfunlist=['relu','leakyrelu'],epoch=1)