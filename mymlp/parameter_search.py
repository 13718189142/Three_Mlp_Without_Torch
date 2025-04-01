import numpy as np
from Structure.model import ThreeLayerNet
from train import train
from test import test


def parameter_search(X_train, y_train, X_val, y_val, X_test, y_test,actfunlist=['relu','relu'],epoch=5):
    
    input_size = X_train.shape[1]
    output_size = 10

    learning_rates = [1e-3, 1e-4]
    hidden_sizes = [(100, 50), (512, 256)]
    reg_strengths = [0,1e-3, 1e-4,1e-5]

    results = {}
    best_val_acc = 0
    best_model = None
    best_model_params = ""
    for lr in learning_rates:
        for hs in hidden_sizes:
            for reg in reg_strengths:
                model = ThreeLayerNet(input_size, hs[0], hs[1], output_size,L2=reg,actfunlist=actfunlist)
                trained_model,_,_,_ = train(model, X_train, y_train, X_val, y_val,
                                      learning_rate=lr,num_epochs=epoch)
                val_acc = test(trained_model, X_val, y_val)
                test_acc = test(trained_model, X_test, y_test)
                results[(lr, hs, reg)] = (val_acc, test_acc)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    trained_model.save()
                    best_model_params = str(lr) + "_" + str(hs) + "_" + str(reg)
                print('lr %e, hidden sizes %s, reg %e; val accuracy: %f test accuracy: %f' % (lr, hs, reg, val_acc, test_acc))

    for lr, hs, reg in sorted(results):
        val_acc, test_acc = results[(lr, hs, reg)]
        print('lr %e, hidden sizes %s, reg %e; val accuracy: %f test accuracy: %f' % (
            lr, hs, reg, val_acc, test_acc))
    print('best model parameters: %s' % best_model_params)
    print('best validation accuracy achieved during cross-validation: %f' % best_val_acc)
    return best_model