import numpy as np
import matplotlib.pyplot as plt
from Structure.model import ThreeLayerNet


def train(model, X_train, y_train, X_val, y_val,
          learning_rate=1e-3, learning_rate_decay=0.95,
          num_epochs=10, batch_size=64, verbose=True):
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    num_iters = num_epochs * iterations_per_epoch

    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(num_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_train_loss = 0
        for i in range(iterations_per_epoch):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            probs = model.forward(X_batch)
            loss = model.loss(y_batch)
            epoch_train_loss += loss

            #model.update(y_batch, learning_rate=learning_rate)
            model.backward(y_batch)
            for i in range(len(model.layers)):
                model.layers[i].weights -= learning_rate * model.layers[i].dweights
                model.layers[i].biases -= learning_rate * model.layers[i].dbiases
            if verbose and (epoch * iterations_per_epoch + i) % 500 == 0:
                print(f'Iteration {epoch * iterations_per_epoch + i} / {num_iters}: loss {loss:.6f}')

        # 计算该 epoch 的平均训练损失
        avg_train_loss = epoch_train_loss / iterations_per_epoch
        train_losses.append(avg_train_loss)

        learning_rate *= learning_rate_decay

        val_probs = model.forward(X_val)
        val_loss = model.loss(y_val)
        val_losses.append(val_loss)

        val_pred = np.argmax(val_probs, axis=1)
        val_acc = np.mean(val_pred == y_val)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if val_acc > 0.90:
            break

    print(f'Best validation accuracy: {best_val_acc:.6f}')
    return model, train_losses, val_losses, val_accuracies