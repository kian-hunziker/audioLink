import sounddevice as sd
import scipy.io.wavfile
from scipy import signal
import numpy as np

class Receiver:
    def __init__(self):
        # sampling frequency
        self.fs = 1 / 800
        # determines how many samples are used to encode one bit
        self.rate = 800
        # modulation frequencies
        self.freq_high = 1 / 80
        self.freq_low = 1 / 100

        self.audioSampleRate = 44100

    def readWav(self, file_name) -> np.ndarray:
        rate, data = scipy.io.wavfile.read(file_name)
        return data

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
        demodulated = np.abs(np.ceil(diff / self.rate))
        return demodulated

    def calculateOffsetToTransmition(self, zeroOne, data):
        #corr = np.correlate(zeroOne, data)

        testCorr = signal.correlate(data, zeroOne, mode="same")
        # lags = signal.correlation_lags(len(data), len(zeroOne))

        # TODO improve this offset calculation
        indices = np.where(testCorr > np.max(testCorr) - 2)
        if len(indices) > 0 and len(indices[0] > 0):
            return indices[0][0]
        else:
            return indices

    def test(self):
        input = self.readWav('test_with_silence.wav')
        #print(self.demodulate(input))
        input = np.concatenate((np.random.rand(1000), input))
        firstChange = self.modulate(self.repencode(np.array([1,0])))
        print(self.calculateOffsetToTransmition(firstChange, input))