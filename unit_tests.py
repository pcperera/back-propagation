import unittest
import numpy as np
import neural_network as nn


class EncoderTest(unittest.TestCase):
    def test_one_hot_encode(self):
        encoding_map = {0: np.array([1, 0, 0, 0]), 1: np.array([0, 1, 0, 0]), 2: np.array([0, 0, 1, 0]), 3: np.array([0, 0, 0, 1])}
        keys_array = np.array(list(encoding_map.keys()))
        encoded_labels = nn.one_hot_encode(keys_array)

        for key, value in encoding_map.items():
            self.assertTrue(np.array_equal(a1=encoded_labels[key], a2=value))

