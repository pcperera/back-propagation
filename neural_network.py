import numpy as np
import matplotlib.pyplot as plt

# Set display precision
np.set_printoptions(precision=16)


class NeuralNetwork:
    """
    Architecture: layer0 (input) --> layer1 --> layer2 --> layer3 (output)
    """

    def __init__(self, x_train: [],  y_train: [], x_test: [], y_test: [], learning_rate=0.1, num_epochs=5000):
        # Initialize data
        self.__x_train = x_train
        self.__y_train = y_train
        self.__x_test = x_test
        self.__y_test = y_test

        # Neural network architecture
        self.__input_size = 14
        self.__layer1_size = 100
        self.__layer2_size = 40
        self.__output_size = 4
        self.__epsilon = 1e-10  # For mathematical stability

        # Initialize training parameters
        self.__learning_rate = learning_rate
        self.__num_epochs = num_epochs

        # Initialize weights and biases
        self.__layer0_to_layer1_weights = np.random.randn(self.__input_size, self.__layer1_size)
        self.__layer1_bias = np.zeros(shape=(1, self.__layer1_size), dtype=int)
        self.__layer1_to_layer2_weights = np.random.randn(self.__layer1_size, self.__layer2_size)
        self.__layer_2_bias = np.zeros(shape=(1, self.__layer2_size), dtype=int)
        self.__layer2_to_layer3_weights = np.random.randn(self.__layer2_size, self.__output_size)
        self.__output_bias = np.zeros(shape=(1, self.__output_size), dtype=int)

        # Initializes loss array
        self.__training_losses = []
        self.__testing_losses = []

        # Initialize accuracy array
        self.__training_accuracies = []
        self.__testing_accuracies = []

    def __forward(self, x: []):
        # Forward propagation
        layer1_input = np.dot(x, self.__layer0_to_layer1_weights) + self.__layer1_bias
        layer1_output = self.__relu(layer1_input)
        layer2_input = np.dot(layer1_output, self.__layer1_to_layer2_weights) + self.__layer_2_bias
        layer2_output = self.__relu(layer2_input)
        output_input = np.dot(layer2_output, self.__layer2_to_layer3_weights) + self.__output_bias
        output = self.__softmax(output_input)
        return layer1_input, layer1_output, layer2_input, layer2_output, output

    def __get_cross_entropy_loss(self, x: [], y: [], output):
        # One-hot-encode and calculate loss using cross-entropy loss
        one_hot_labels = self.one_hot_encode(self.__output_size, y)  # Convert Y to one-hot vectors
        cross_entropy_loss = -np.sum(one_hot_labels * np.log(output + self.__epsilon)) / len(x)
        return cross_entropy_loss, one_hot_labels

    def train(self):
        for epoch in range(1, self.__num_epochs + 1):
            layer1_input, layer1_output, layer2_input, layer2_output, output = self.__forward(x=self.__x_train)
            training_cross_entropy_loss, training_one_hot_labels = self.__get_cross_entropy_loss(x=self.__x_train, y=self.__y_train, output=output)
            self.__training_losses.append(training_cross_entropy_loss)

            # Backpropagation
            output_error = output - training_one_hot_labels
            layer2_error = np.dot(output_error, self.__layer2_to_layer3_weights.T)
            layer2_error[layer2_input <= 0] = 0
            layer1_error = np.dot(layer2_error, self.__layer1_to_layer2_weights.T)
            layer1_error[layer1_input <= 0] = 0

            # Compute gradients
            grad_weights_layer2_output = np.dot(layer2_output.T, output_error)
            grad_bias_output = np.sum(output_error, axis=0, keepdims=True)
            grad_weights_layer1_layer2 = np.dot(layer1_output.T, layer2_error)
            grad_bias_layer2 = np.sum(layer2_error, axis=0, keepdims=True)
            grad_weights_input_layer1 = np.dot(self.__x_train.T, layer1_error)
            grad_bias_layer1 = np.sum(layer1_error, axis=0, keepdims=True)

            # Update weights and biases
            self.__layer0_to_layer1_weights = self.__layer0_to_layer1_weights - self.__learning_rate * grad_weights_input_layer1
            self.__layer1_bias = self.__layer1_bias - self.__learning_rate * grad_bias_layer1
            self.__layer1_to_layer2_weights = self.__layer1_to_layer2_weights - self.__learning_rate * grad_weights_layer1_layer2
            self.__layer_2_bias = self.__layer_2_bias - self.__learning_rate * grad_bias_layer2
            self.__layer2_to_layer3_weights = self.__layer2_to_layer3_weights - self.__learning_rate * grad_weights_layer2_output
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

    # Activation functions
    @staticmethod
    def __relu(x):
        return np.maximum(0, x)

    @staticmethod
    def __softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def one_hot_encode(num_classes, labels: []):
        num_samples = len(labels)
        one_hot_encoding_labels = np.zeros((num_samples, num_classes), dtype=int)
        for sample in range(num_samples):
            one_hot_encoding_labels[sample, labels[sample]] = 1
        return one_hot_encoding_labels
