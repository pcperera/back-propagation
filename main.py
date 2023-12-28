import numpy as np
import pandas as pd
from neural_network import NeuralNetwork

# learning_rates = [1, 0.1, 0.001]
learning_rates = [0.001]
num_epochs = 1000

if __name__ == "__main__":
    data_directory = "Task_2_data"

    x_train = pd.read_csv(f"{data_directory}/x_train.csv")
    y_train = pd.read_csv(f"{data_directory}/y_train.csv")

    x_test = pd.read_csv(f"{data_directory}/x_test.csv")
    y_test = pd.read_csv(f"{data_directory}/y_test.csv")

    for learning_rate in learning_rates:
        print(f"Training neural network with learning rate: {learning_rate}, number of epochs: {num_epochs}")
        nn = NeuralNetwork(x_train=x_train.values, y_train=y_train.values, x_test=x_test.values, y_test=y_test.values, learning_rate=learning_rate, num_epochs=num_epochs)
        nn.train()
        nn.plot()

        # Test from the data poit given in data_point.txt.
        predicted_labels = nn.predict(x=[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1])
        assert np.array_equal(predicted_labels, [3])
