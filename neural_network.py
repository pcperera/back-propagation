import numpy as np

# Neural network architecture
input_size = 14
hidden1_size = 100
hidden2_size = 40
output_size = 4
learning_rate = 0.01
epochs = 1000

# X
X = [-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]

# Y
Y = [3]
num_classes = 4  # Number of classes


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


def train():
    # Initialize weights and biases
    input_hidden1_weights = np.random.randn(input_size, hidden1_size)
    hidden1_bias = np.zeros((1, hidden1_size))
    hidden1_hidden2_weights = np.random.randn(hidden1_size, hidden2_size)
    hidden_2_bias = np.zeros((1, hidden2_size))
    hidden2_output_weights = np.random.randn(hidden2_size, output_size)
    output_bias = np.zeros((1, output_size))

    for epoch in range(epochs):
        # Forward propagation
        print(f"Epoch {epoch}")
        hidden1_input = np.dot(X, input_hidden1_weights) + hidden1_bias
        hidden1_output = relu(hidden1_input)
        hidden2_input = np.dot(hidden1_output, hidden1_hidden2_weights) + hidden_2_bias
        hidden2_output = relu(hidden2_input)
        output_input = np.dot(hidden2_output, hidden2_output_weights) + output_bias
        output = softmax(output_input)

        # Calculate loss using cross-entropy
        one_hot_labels = one_hot_encode(Y)  # Convert Y to one-hot vectors
        print(one_hot_labels)

        # Compute loss
        loss = -np.sum(one_hot_labels * np.log(output)) / len(X)

        # Backpropagation
        output_error = output - one_hot_labels
        hidden2_error = np.dot(output_error, hidden2_output_weights.T)
        hidden2_error[hidden2_input <= 0] = 0
        hidden1_error = np.dot(hidden2_error, hidden1_hidden2_weights.T)
        hidden1_error[hidden1_input <= 0] = 0

        # Compute gradients
        grad_weights_hidden2_output = np.dot(hidden2_output.T, output_error)
        grad_bias_output = np.sum(output_error, axis=0, keepdims=True)
        grad_weights_hidden1_hidden2 = np.dot(hidden1_output.T, hidden2_error)
        grad_bias_hidden2 = np.sum(hidden2_error, axis=0, keepdims=True)
        grad_weights_input_hidden1 = np.dot(X.T, hidden1_error)
        grad_bias_hidden1 = np.sum(hidden1_error, axis=0, keepdims=True)

        # Update weights and biases
        input_hidden1_weights -= learning_rate * grad_weights_input_hidden1
        hidden1_bias -= learning_rate * grad_bias_hidden1
        hidden1_hidden2_weights -= learning_rate * grad_weights_hidden1_hidden2
        hidden_2_bias -= learning_rate * grad_bias_hidden2
        hidden2_output_weights -= learning_rate * grad_weights_hidden2_output
        output_bias -= learning_rate * grad_bias_output

        # Print loss
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

