import unittest
import numpy as np
from neural_network import NeuralNetwork


class EncoderTest(unittest.TestCase):
    def test_one_hot_encode(self):
        encoding_map = {0: np.array([1, 0, 0, 0]), 1: np.array([0, 1, 0, 0]), 2: np.array([0, 0, 1, 0]), 3: np.array([0, 0, 0, 1])}
        keys_array = np.array(list(encoding_map.keys()))
        encoded_labels = NeuralNetwork.one_hot_encode(num_classes=len(encoding_map.keys()), labels=keys_array)

        for key, value in encoding_map.items():
            self.assertTrue(np.array_equal(a1=encoded_labels[key], a2=value))
