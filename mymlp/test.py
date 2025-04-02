import numpy as np


def test(model, X_test, y_test):
    probs = model.forward(X_test)
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred == y_test)
    return acc