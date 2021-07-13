import numpy as np
import simpleaudio as sa
import scipy.io
import scipy.io.wavfile

class Hamming:
    def __init__(self):
        self.G = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]])
        self.R = np.array([[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])

    def encode(self, input):
        return np.dot(self.G, input) % 2

    def parityCheck(self, input):
        return np.sum(np.dot(self.H, input) % 2) == 0

    def errorPositionVector(self, input):
        return np.dot(self.H, input) % 2

    def decimalErrorPosition(self, input):
        errorVector = self.errorPositionVector(input)
        res = 4 * errorVector[2] + 2 * errorVector[1] + errorVector[0]
        return res - 1

    def decode(self, input):
        return np.dot(self.R, input)

    def correct(self, input):
        if not self.parityCheck(input):
            errorPos = self.decimalErrorPosition(input)
            if input[errorPos] == 1:
                input[errorPos] = 0
            else:
                input[errorPos] = 1
        return input




hamming = Hamming()
i = np.array([1, 0, 1, 1])