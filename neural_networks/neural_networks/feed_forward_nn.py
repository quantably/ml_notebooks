import matplotlib.pyplot as plt
from backward_prop import *
from forward_prop import *
import h5py
import numpy as np

np.random.seed(1)


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def accuracy(pred, actual):
    acc = 100 - np.sum(np.abs(actual - pred)) * 100 / pred.shape[1]
    return acc


def predict(X, parameters):
    AL, _ = L_model_forward(X, parameters)
    return AL > 0.5


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    print(f"Training deep neural network model with {len(layers_dims)} hidden layers")
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        if i % 100 == 0 and print_cost:
            print(f"Cost after {i} iterations: {cost}")
            costs.append(cost)

        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)

    print("Model training completed")
    print("Plotting cost function")
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x_flatten = train_set_x_flatten/255.
test_set_x_flatten = test_set_x_flatten/255.

#layers_dims = (12288, 7, 1)
layers_dims = (12288, 20, 7, 5, 1)
parameters = L_layer_model(train_set_x_flatten, train_set_y, layers_dims, num_iterations = 2500, print_cost = True)

predictions_train = predict(train_set_x_flatten, parameters)
predictions_test = predict(test_set_x_flatten, parameters)

train_acc = accuracy(predictions_train, train_set_y)
test_acc = accuracy(predictions_test, test_set_y)

print()
print(f"Training Accuracy: {train_acc}%")
print(f"Testing Accuracy: {test_acc}%")