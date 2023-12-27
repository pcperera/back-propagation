import numpy as np

# Neural network architecture
input_size = 14
hidden1_size = 100
hidden2_size = 40
output_size = 4


num_classes = 4


# Activation functions
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def one_hot_encode(labels: []):
    num_samples = len(labels)
    one_hot_encoding_labels = np.zeros((num_samples, num_classes), dtype=int)
    for sample in range(num_samples):
        one_hot_encoding_labels[sample, labels[sample]] = 1
    return one_hot_encoding_labels


class NeuralNetwork:

    def __init__(self, x_train: [],  y_train: [], learning_rate=0.1, num_epochs=1000):
        # Initialize training data
        self.__x_train = x_train
        self.__y_train = y_train

        # Initialize training parameters
        self.__learning_rate = learning_rate
        self.__num_epochs = num_epochs

        # Initialize weights and biases
        self.__input_hidden1_weights = np.random.randn(input_size, hidden1_size)
        self.__hidden1_bias = np.zeros((1, hidden1_size))
        self.__hidden1_hidden2_weights = np.random.randn(hidden1_size, hidden2_size)
        self.__hidden_2_bias = np.zeros((1, hidden2_size))
        self.__hidden2_output_weights = np.random.randn(hidden2_size, output_size)
        self.__output_bias = np.zeros((1, output_size))

    def train(self):
        for epoch in range(self.__num_epochs):
            # Forward propagation
            print(f"Epoch {epoch}")
            hidden1_input = np.dot(self.__x_train, self.__input_hidden1_weights) + self.__hidden1_bias
            hidden1_output = relu(hidden1_input)
            hidden2_input = np.dot(hidden1_output, self.__hidden1_hidden2_weights) + self.__hidden_2_bias
            hidden2_output = relu(hidden2_input)
            output_input = np.dot(hidden2_output, self.__hidden2_output_weights) + self.__output_bias
            output = softmax(output_input)

            # Calculate loss using cross-entropy
            one_hot_labels = one_hot_encode(self.__y_train)  # Convert Y to one-hot vectors

            # Compute loss
            loss = -np.sum(one_hot_labels * np.log(output)) / len(self.__x_train)

            # Backpropagation
            output_error = output - one_hot_labels
            hidden2_error = np.dot(output_error, self.__hidden2_output_weights.T)
            hidden2_error[hidden2_input <= 0] = 0
            hidden1_error = np.dot(hidden2_error, self.__hidden1_hidden2_weights.T)
            hidden1_error[hidden1_input <= 0] = 0

            # Compute gradients
            grad_weights_hidden2_output = np.dot(hidden2_output.T, output_error)
            grad_bias_output = np.sum(output_error, axis=0, keepdims=True)
            grad_weights_hidden1_hidden2 = np.dot(hidden1_output.T, hidden2_error)
            grad_bias_hidden2 = np.sum(hidden2_error, axis=0, keepdims=True)
            grad_weights_input_hidden1 = np.dot(self.__x_train.T, hidden1_error)
            grad_bias_hidden1 = np.sum(hidden1_error, axis=0, keepdims=True)

            # Update weights and biases
            self.__input_hidden1_weights = self.__input_hidden1_weights - self.__learning_rate * grad_weights_input_hidden1
            self.__hidden1_bias = self.__hidden1_bias - self.__learning_rate * grad_bias_hidden1
            self.__hidden1_hidden2_weights = self.__hidden1_hidden2_weights - self.__learning_rate * grad_weights_hidden1_hidden2
            self.__hidden_2_bias = self.__hidden_2_bias - self.__learning_rate * grad_bias_hidden2
            self.__hidden2_output_weights = self.__hidden2_output_weights - self.__learning_rate * grad_weights_hidden2_output
            self.__output_bias = self.__output_bias - self.__learning_rate * grad_bias_output

            # Print loss
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

