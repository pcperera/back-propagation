import numpy as np

# Set display precision
np.set_printoptions(precision=16)


# Activation functions
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exponentials = np.exp(x - np.max(x, axis=1, keepdims=True))
    softmax_output = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    return one_hot_encode(num_classes=4, labels=np.argmax(softmax_output, axis=1))


def one_hot_encode(num_classes, labels: []):
    num_samples = len(labels)
    one_hot_encoding_labels = np.zeros((num_samples, num_classes), dtype=int)
    for sample in range(num_samples):
        one_hot_encoding_labels[sample, labels[sample]] = 1
    return one_hot_encoding_labels


def decode(output):
    return np.argmax(output, axis=1)


def cross_entropy_loss(x_true: [], y_true: [], y_predicted, epsilon):
    # One-hot-encode and calculate loss using cross-entropy loss
    loss = -np.sum(y_true * np.log(np.clip(y_predicted, epsilon, 1 - epsilon))) / len(x_true)
    return loss
