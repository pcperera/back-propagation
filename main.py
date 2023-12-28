import numpy as np
import pandas as pd
from neural_network import NeuralNetwork

x_data_point = np.array([-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]).reshape(1, -1)
y_data_point = np.array([3]).reshape(1, -1)


def task_1():
    print("Running Task_1")
    data_directory = "Task_1_original/a"
    file_suffix = ".csv"  # "-100-40-4.csv"
    biases = pd.read_csv(f"{data_directory}/b{file_suffix}", header=None)
    biases.drop(biases.columns[0], axis=1, inplace=True)
    weights = pd.read_csv(f"{data_directory}/w{file_suffix}", header=None)
    weights.drop(weights.columns[0], axis=1, inplace=True)
    nn = NeuralNetwork(x_train=x_data_point, y_train=y_data_point, x_test=None, y_test=None, weights=weights.values, biases=biases.values, num_epochs=1)
    nn.train(log_derivatives=True)


def task_2():
    print("Running Task_2")
    learning_rates = [1, 0.1, 0.001, 0.0001]
    num_epochs = 2000

    data_directory = "Task_2_data"

    x_train = pd.read_csv(f"{data_directory}/x_train.csv")
    y_train = pd.read_csv(f"{data_directory}/y_train.csv")

    x_test = pd.read_csv(f"{data_directory}/x_test.csv")
    y_test = pd.read_csv(f"{data_directory}/y_test.csv")

    for learning_rate in learning_rates:
        print(f"Training neural network with learning rate: {learning_rate}, number of epochs: {num_epochs}")
        nn = NeuralNetwork(x_train=x_train.values, y_train=y_train.values, x_test=x_test.values, y_test=y_test.values, learning_rate=learning_rate, num_epochs=num_epochs)
        nn.train(log_derivatives=False)
        nn.plot()

        # Test using the data point given in data_point.txt.
        predicted_labels = nn.predict(x=x_data_point)
        result = np.array_equal(predicted_labels, y_data_point)
        print(f"Result of validation of test data point: {result}")


if __name__ == "__main__":
    task_1()
    # task_2()

