import numpy as np

# Set display precision
np.set_printoptions(precision=16)


# Activation functions
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exponential_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    softmax_probabilities = exponential_scores / np.sum(exponential_scores, axis=1, keepdims=True)
    return one_hot_encode(num_classes=4, labels=np.argmax(softmax_probabilities, axis=1))


def one_hot_encode(num_classes, labels: []):
    num_samples = len(labels)
    one_hot_encoding_labels = np.zeros((num_samples, num_classes), dtype=int)
    for sample in range(num_samples):
        one_hot_encoding_labels[sample, labels[sample]] = 1
    return one_hot_encoding_labels


def decode(output):
    return np.argmax(output, axis=1)


def cross_entropy_loss(y_true: [], y_predicted, epsilon=1e-64):
    # One-hot-encode and calculate loss using cross-entropy loss
    loss = -np.sum(y_true * np.log(np.clip(y_predicted, epsilon, 1 - epsilon))) / len(y_true)
    return loss


def xavier_initialization(n_in, n_out):
    variance = 2.0 / (n_in + n_out) * 10
    stddev = np.sqrt(variance)
    return np.random.randn(n_in, n_out) * stddev


def he_initialization(n_in, n_out):
    variance = 2.0 / n_in
    stddev = np.sqrt(variance)
    return np.random.randn(n_in, n_out) * stddev
