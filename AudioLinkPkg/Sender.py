

import numpy as np
import simpleaudio as sa
import scipy.io



class Sender:

    def __init__(self):
        # sampling frequency
        self.fs = 1/800
        # determines how many samples are used to encode one bit
        self.rate = 800
        # modulation frequencies
        self.freq_high = 1/ 8
        self.freq_low = 1/ 20

    def playAudio(self, data):
        audio = data * (2 ** 15 - 1) / np.max(np.abs(data))

        audio = audio.astype(np.int16)
        play_onj = sa.play_buffer(audio, 1, 2, 44100)

        play_onj.wait_done()


    def getTestTone(self):
        frequency = 440
        seconds = 3

        t = np.linspace(0, seconds, seconds * self.fs, False)

        note = np.sin(frequency * t * 2 *np.pi)

        return note

    def getTestDataAsBits(self):
        s = np.array([1, 0, 1, 1, 0, 0, 1])
        return s

    def repencode(self, data):
        encoded = np.repeat(data, self.rate)
        return encoded

    def modulate(self, data):
        length = len(data)
        t = np.linspace(0, length, length)
        mod_high = np.multiply(np.sin(self.freq_high * t * 2 * np.pi), data)
        mod_low = np.multiply(np.sin(self.freq_low * t * 2 * np.pi), 1 - data)
        return mod_high + mod_low


    def demodulate(self, data):
        t = np.linspace(0, 1 / self.fs, self.rate)

        sin_high = np.sin(self.freq_high * t * 2 * np.pi)
        sin_low = np.sin(self.freq_low * t * 2 * np.pi)

        data_matrix = np.reshape(data, (len(data) // self.rate, self.rate))
        sol_high = np.dot(sin_high, np.transpose(data_matrix))
        sol_low = np.dot(sin_low, np.transpose(data_matrix))

        diff = sol_high - sol_low

    def test(self):
        data = self.getTestDataAsBits()
        encoded = self.repencode(data)
        self.demodulate(self.modulate(encoded))
        self.playAudio(self.modulate(encoded))




