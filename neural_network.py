import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from neural_network_utils import relu, softmax, one_hot_encode, decode, cross_entropy_loss, xavier_initialization, he_initialization

# Set display precision
np.set_printoptions(precision=16)


class NeuralNetwork:
    """
    Architecture: layer0 (input) --> layer1 --> layer2 --> layer3 (output)
    """

    def __init__(self, x_train: [],  y_train: [], x_test: [], y_test: [], learning_rate=0.1, num_epochs=5000, weights=None, biases=None):
        # Initialize data
        self.__x_train = x_train
        self.__y_train = y_train
        self.__x_test = x_test
        self.__y_test = y_test

        # Neural network architecture
        self.__layer0_size = 14  # Input size
        self.__layer1_size = 100
        self.__layer2_size = 40
        self.__layer3_size = 4  # Output size
        self.__epsilon = 1e-64  # For mathematical stability

        # Initialize training parameters
        self.__learning_rate = learning_rate
        self.__num_epochs = num_epochs

        # Initialize weights and biases
        if weights is None:
            self.__layer0_to_layer1_weights = xavier_initialization(self.__layer0_size, self.__layer1_size)
            self.__layer1_to_layer2_weights = xavier_initialization(self.__layer1_size, self.__layer2_size)
            self.__layer2_to_layer3_weights = xavier_initialization(self.__layer2_size, self.__layer3_size)
        else:
            self.__layer0_to_layer1_weights = weights[0:14, :100]
            self.__layer1_to_layer2_weights = weights[14:114, :40]
            self.__layer2_to_layer3_weights = weights[114:154, :4]

        if biases is None:
            self.__layer1_bias = np.zeros(shape=(1, self.__layer1_size))
            self.__layer2_bias = np.zeros(shape=(1, self.__layer2_size))
            self.__layer3_bias = np.zeros(shape=(1, self.__layer3_size))
        else:
            self.__layer1_bias = biases[0:1, :100]
            self.__layer2_bias = biases[1:2, :40]
            self.__layer3_bias = biases[2:3, :4]

        # Initializes loss array
        self.__training_losses = []
        self.__testing_losses = []

        # Initialize accuracy array
        self.__training_accuracies = []
        self.__testing_accuracies = []

    def __forward(self, x: []):
        # Forward propagation
        layer1_input = np.dot(x, self.__layer0_to_layer1_weights) + self.__layer1_bias
        layer1_output = relu(layer1_input)
        layer2_input = np.dot(layer1_output, self.__layer1_to_layer2_weights) + self.__layer2_bias
        layer2_output = relu(layer2_input)
        layer3_input = np.dot(layer2_output, self.__layer2_to_layer3_weights) + self.__layer3_bias
        layer3_output = softmax(layer3_input)
        return layer1_input, layer1_output, layer2_input, layer2_output, layer3_output

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

    def __run(self, train_mode: bool= True):
        max_epochs = self.__num_epochs if train_mode else 1

        for epoch in range(1, max_epochs + 1):
            # Shuffle training data before each epoch
            training_shuffled_indices = np.random.permutation(len(self.__x_train))
            x_train_shuffled = self.__x_train[training_shuffled_indices]
            y_train_shuffled = self.__y_train[training_shuffled_indices]

            layer1_input, layer1_output, layer2_input, layer2_output, layer3_output = self.__forward(x=x_train_shuffled)
            y_training_encoded = one_hot_encode(self.__layer3_size, y_train_shuffled)  # Convert Y to one-hot vectors
            training_cross_entropy_loss = cross_entropy_loss(x_true=x_train_shuffled, y_true=y_training_encoded, y_predicted=layer3_output, epsilon=self.__epsilon)
            self.__training_losses.append(training_cross_entropy_loss)

            # Backpropagation
            layer3_error = layer3_output - y_training_encoded
            layer2_error = np.dot(layer3_error, self.__layer2_to_layer3_weights.T)
            layer2_error[layer2_input <= 0] = 0
            layer1_error = np.dot(layer2_error, self.__layer1_to_layer2_weights.T)
            layer1_error[layer1_input <= 0] = 0

            # Compute derivatives (gradients)
            layer2_to_layer3_derivative_of_weights = np.dot(layer2_output.T, layer3_error)
            layer3_derivative_of_bias = np.sum(layer3_error, axis=0, keepdims=True)
            layer1_to_layer2_derivative_of_weights = np.dot(layer1_output.T, layer2_error)
            layer2_derivative_of_bias = np.sum(layer2_error, axis=0, keepdims=True)
            layer0_to_layer1_derivative_of_weights = np.dot(x_train_shuffled.T, layer1_error)
            layer1_derivative_of_bias = np.sum(layer1_error, axis=0, keepdims=True)

            # Update weights and biases
            self.__layer0_to_layer1_weights = self.__layer0_to_layer1_weights - (self.__learning_rate * layer0_to_layer1_derivative_of_weights)
            self.__layer1_bias = self.__layer1_bias - (self.__learning_rate * layer1_derivative_of_bias)
            self.__layer1_to_layer2_weights = self.__layer1_to_layer2_weights - (self.__learning_rate * layer1_to_layer2_derivative_of_weights)
            self.__layer2_bias = self.__layer2_bias - (self.__learning_rate * layer2_derivative_of_bias)
            self.__layer2_to_layer3_weights = self.__layer2_to_layer3_weights - (self.__learning_rate * layer2_to_layer3_derivative_of_weights)
            self.__layer3_bias = self.__layer3_bias - (self.__learning_rate * layer3_derivative_of_bias)

            # Training data metrics
            training_predictions = decode(layer3_output)  # Predicted classes for training data
            training_true_labels = decode(y_training_encoded)  # True classes for training data
            training_accuracy = np.mean(training_predictions == training_true_labels)  # Training accuracy
            self.__training_accuracies.append(training_accuracy)

            # Testing data metrics
            is_test_metrics_enabled = self.__x_test is not None and self.__y_test is not None
            if is_test_metrics_enabled:
                testing_shuffled_indices = np.random.permutation(len(self.__x_test))
                x_test_shuffled = self.__x_test[testing_shuffled_indices]
                y_test_shuffled = self.__y_test[testing_shuffled_indices]

                _, _, _, _, testing_output = self.__forward(x=x_test_shuffled)
                y_testing_encoded = one_hot_encode(self.__layer3_size, y_test_shuffled)
                testing_cross_entropy_loss = cross_entropy_loss(x_true=x_test_shuffled, y_true=y_testing_encoded, y_predicted=testing_output, epsilon=self.__epsilon)
                self.__testing_losses.append(testing_cross_entropy_loss)
                testing_predictions = decode(testing_output)  # Predicted classes for testing data
                testing_true_labels = decode(y_testing_encoded)  # True classes for testing data

                # Calculate testing accuracy
                testing_accuracy = np.mean(testing_predictions == testing_true_labels)
                self.__testing_accuracies.append(testing_accuracy)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Training Cross Entropy Loss: {training_cross_entropy_loss}, Training Accuracy: {training_accuracy}")
                if is_test_metrics_enabled:
                    print(f"Epoch {epoch}, Testing Cross Entropy Loss: {testing_cross_entropy_loss}, Testing Accuracy: {testing_accuracy}")

            if (not train_mode) and epoch == self.__num_epochs:
                task_1_directory = "Task_1"
                os.makedirs(name=task_1_directory, exist_ok=True)

                # Derivative of bias
                file_name = f"{task_1_directory}/db.csv"
                print(f"Saving bias derivatives to {file_name}")
                with open(file_name, 'w') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(layer1_derivative_of_bias)
                    csv_writer.writerows(layer2_derivative_of_bias)
                    csv_writer.writerows(layer3_derivative_of_bias)

                # Derivative of weights
                file_name = f"{task_1_directory}/dw.csv"
                print(f"Saving weight derivatives to {file_name}")
                with open(file_name, 'w') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(layer0_to_layer1_derivative_of_weights)
                    csv_writer.writerows(layer1_to_layer2_derivative_of_weights)
                    csv_writer.writerows(layer2_to_layer3_derivative_of_weights)

    def train(self):
        self.__run(train_mode=True)

    def calculate_derivatives(self):
        self.__run(train_mode=False)

    def plot(self):
        y_label = "Cross Entropy Loss"
        training_mode = "Training"
        testing_mode = "Testing"
        self.__plot_internal(mode=training_mode, y=self.__training_losses, y_label=y_label)
        self.__plot_internal(mode=testing_mode, y=self.__testing_losses, y_label=y_label)

        y_label = "Accuracy"
        self.__plot_internal(mode=training_mode, y=self.__training_accuracies, y_label=y_label)
        self.__plot_internal(mode=testing_mode, y=self.__testing_accuracies, y_label=y_label)

    def predict(self, x: []):
        _, _, _, _, predicted_output = self.__forward(x=x)
        return decode(predicted_output)

