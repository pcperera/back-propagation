import numpy as np

# Set display precision
np.set_printoptions(precision=16)


# Activation functions
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exponentials = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def one_hot_encode(num_classes, labels: []):
    num_samples = len(labels)
    one_hot_encoding_labels = np.zeros((num_samples, num_classes), dtype=int)
    for sample in range(num_samples):
        one_hot_encoding_labels[sample, labels[sample]] = 1
    return one_hot_encoding_labels
