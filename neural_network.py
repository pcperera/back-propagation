import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, x_train: [],  y_train: [], x_test: [], y_test: [], learning_rate=0.1, num_epochs=5000):
        # Initialize data
        self.__x_train = x_train
        self.__y_train = y_train
        self.__x_test = x_test
        self.__y_test = y_test

        # Neural network architecture
        self.__input_size = 14
        self.__hidden1_size = 100
        self.__hidden2_size = 40
        self.__output_size = 4
        self.__epsilon = 1e-10

        # Initialize training parameters
        self.__learning_rate = learning_rate
        self.__num_epochs = num_epochs

        # Initialize weights and biases
        self.__input_hidden1_weights = np.random.randn(self.__input_size, self.__hidden1_size)
        self.__hidden1_bias = np.zeros(shape=(1, self.__hidden1_size), dtype=int)
        self.__hidden1_hidden2_weights = np.random.randn(self.__hidden1_size, self.__hidden2_size)
        self.__hidden_2_bias = np.zeros(shape=(1, self.__hidden2_size), dtype=int)
        self.__hidden2_output_weights = np.random.randn(self.__hidden2_size, self.__output_size)
        self.__output_bias = np.zeros(shape=(1, self.__output_size), dtype=int)

        # Initializes loss array
        self.__training_losses = []
        self.__testing_losses = []

        # Initialize accuracy array
        self.__training_accuracies = []
        self.__testing_accuracies = []

    def __forward(self, x: []):
        # Forward propagation
        hidden1_input = np.dot(x, self.__input_hidden1_weights) + self.__hidden1_bias
        hidden1_output = relu(hidden1_input)
        hidden2_input = np.dot(hidden1_output, self.__hidden1_hidden2_weights) + self.__hidden_2_bias
        hidden2_output = relu(hidden2_input)
        output_input = np.dot(hidden2_output, self.__hidden2_output_weights) + self.__output_bias
        output = softmax(output_input)
        output = np.clip(output, self.__epsilon, 1 - self.__epsilon)
        return hidden1_input, hidden1_output, hidden2_input, hidden2_output, output

    def __get_cross_entropy_loss(self, x: [], y: [], output):
        # One-hot-encode
        one_hot_labels = one_hot_encode(y)  # Convert Y to one-hot vectors

        # Calculate loss using cross-entropy loss
        cross_entropy_loss = -np.sum(one_hot_labels * np.log(output)) / len(x)
        return cross_entropy_loss, one_hot_labels

    def train(self):
        for epoch in range(1, self.__num_epochs + 1):
            hidden1_input, hidden1_output, hidden2_input, hidden2_output, output = self.__forward(x=self.__x_train)
            training_cross_entropy_loss, training_one_hot_labels = self.__get_cross_entropy_loss(x=self.__x_train, y=self.__y_train, output=output)
            self.__training_losses.append(training_cross_entropy_loss)

            # Backpropagation
            output_error = output - training_one_hot_labels
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

            # Training data metrics
            training_predictions = np.argmax(output, axis=1)  # Predicted classes for training data
            training_true_labels = np.argmax(training_one_hot_labels, axis=1)  # True classes for training data
            training_accuracy = np.mean(training_predictions == training_true_labels)  # Training accuracy
            self.__training_accuracies.append(training_accuracy)

            # Testing data metrics
            _, _, _, _, testing_output = self.__forward(x=self.__x_test)
            testing_cross_entropy_loss, testing_one_hot_labels = self.__get_cross_entropy_loss(x=self.__x_test, y=self.__y_test, output=testing_output)
            self.__testing_losses.append(testing_cross_entropy_loss)
            testing_predictions = np.argmax(testing_output, axis=1)  # Predicted classes for testing data
            testing_true_labels = np.argmax(testing_one_hot_labels, axis=1)  # True classes for testing data
        
            # Calculate testing accuracy
            test_accuracy = np.mean(testing_predictions == testing_true_labels)
            self.__testing_accuracies.append(test_accuracy)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Training Cross Entropy Loss: {training_cross_entropy_loss}, Training Accuracy: {training_accuracy}")
                print(f"Epoch {epoch}, Testing Cross Entropy Loss: {testing_cross_entropy_loss}, Testing Accuracy: {test_accuracy}")

    def plot(self):
        y_label = "Cross Entropy Loss"
        training_mode = "Training"
        testing_mode = "Testing"
        self.__plot_internal(mode=training_mode, y=self.__training_losses, y_label=y_label)
        self.__plot_internal(mode=testing_mode, y=self.__testing_losses, y_label=y_label)

        y_label = "Accuracy"
        self.__plot_internal(mode=training_mode, y=self.__training_accuracies, y_label=y_label)
        self.__plot_internal(mode=testing_mode, y=self.__testing_accuracies, y_label=y_label)

    def __plot_internal(self, mode: str, y, y_label: str):
        plt.figure(figsize=(20, 8))
        plt.plot(range(self.__num_epochs), y, label=f"{mode} {y_label}")
        plt.xlabel("Epoch")
        plt.ylabel(y_label)
        plt.title(f"{mode} {y_label} over Epochs (Learning Rate: {self.__learning_rate})")
        plt.legend()
        file_name = f"lr {self.__learning_rate} {mode} {y_label}.png"
        file_name = file_name.replace(" ", "_").lower()
        plt.savefig(file_name)
        plt.close()
