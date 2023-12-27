import pandas as pd
from neural_network import NeuralNetwork

learning_rates = [1, 0.1, 0.001]

if __name__ == "__main__":
    x_train = pd.read_csv("Task_2/x_train.csv")
    y_train = pd.read_csv("Task_2/y_train.csv")

    x_test = pd.read_csv("Task_2/x_test.csv")
    y_test = pd.read_csv("Task_2/y_test.csv")

    nn = NeuralNetwork(x_train=x_train.values, y_train=y_train.values, learning_rate=0.1)
    nn.train()