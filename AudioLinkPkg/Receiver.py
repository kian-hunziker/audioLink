import sounddevice as sd
import scipy.io.wavfile
from scipy import signal
import numpy as np
import simpleaudio as sa

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

        self.pilot1 = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1,
                                1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1,
                                0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
                                0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
                                0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=np.uint8)

        self.pilot2 = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1,
                                0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0,
                                0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1,
                                0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1,
                                1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0], dtype=np.uint8)

    def readWav(self, file_name) -> np.ndarray:
        rate, data = scipy.io.wavfile.read(file_name)

        if data.dtype == np.int16:
            return data.astype(np.float32, order='C') / 32768.0
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

    def truncateToTauS(self, data, offset):
        truncated_start = data[(offset % self.rate):]
        truncated = truncated_start[:len(truncated_start) - (len(truncated_start) % self.rate)]
        return truncated

    def convertToOneMinusOne(self, data):
        return 2 * data - 1

    def removePilots(self, data):
        pilot_1_converted = self.convertToOneMinusOne(self.pilot1.astype(np.float32))
        pilot_2_converted = self.convertToOneMinusOne(self.pilot2.astype(np.float32))

        offset_1 = self.calculateOffsetToTransmition(pilot_1_converted, self.convertToOneMinusOne(data)) - 50
        trunc_1 = data[offset_1 + len(self.pilot1):]
        offset_2 = self.calculateOffsetToTransmition(pilot_2_converted, self.convertToOneMinusOne(trunc_1)) - 50
        trunc_2 = trunc_1[:offset_2]
        return trunc_2


    def test(self):
        input = self.readWav('test_ableton_with_noise_quiet.wav')
        #print(self.demodulate(input))
        #input = np.concatenate((np.random.rand(1000), input, np.random.rand(1000)))
        firstChange = self.modulate(self.repencode(np.array([1,0])))
        offset = self.calculateOffsetToTransmition(firstChange, input)

        truncated = self.truncateToTauS(input, offset)
        #truncated = input
        decoded = self.demodulate(truncated)
        actual = self.removePilots(decoded)

        expected = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
        diff = expected - actual
        error_sum = np.sum(np.abs(diff))

        audio = input * (2 ** 15 - 1) / np.max(np.abs(input))

        audio = audio.astype(np.int16)
        play_onj = sa.play_buffer(audio, 1, 2, self.audioSampleRate)

        play_onj.wait_done()

        print('actual: ', actual)
        print('difference: ', diff)
        print('error sum ', error_sum)