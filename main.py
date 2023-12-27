import pandas as pd
from neural_network import NeuralNetwork

learning_rates = [1, 0.1, 0.001]
num_epochs = 1000

if __name__ == "__main__":
    x_train = pd.read_csv("Task_2/x_train.csv")
    y_train = pd.read_csv("Task_2/y_train.csv")

    x_test = pd.read_csv("Task_2/x_test.csv")
    y_test = pd.read_csv("Task_2/y_test.csv")

    for learning_rate in learning_rates:
        print(f"Training neural network with learning rate: {learning_rate}, number of epochs: {num_epochs}")
        nn = NeuralNetwork(x_train=x_train.values, y_train=y_train.values, learning_rate=learning_rate, num_epochs=num_epochs)
        nn.train()
        nn.plot()