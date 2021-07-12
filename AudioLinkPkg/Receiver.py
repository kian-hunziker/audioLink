import sounddevice as sd
import scipy.io.wavfile
from scipy import signal
import numpy as np
import simpleaudio as sa
import sounddevice as sd
from scipy.io.wavfile import write


class Receiver:
    def __init__(self):
        # sampling frequency
        self.fs = 1 / 160
        # determines how many samples are used to encode one bit
        self.rate = 160
        # modulation frequencies
        self.freq_high = 1 / 16
        self.freq_low = 1 / 20

        self.f3 = 1 / 40
        self.f4 = 1 / 8

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

    def repdecode(self, data, n):
        try:
            padding = len(data) % n
            if padding > 0:
                print('len', len(data))
                print('padding:', padding)
                data = np.concatenate((data, np.zeros(n - padding)))
            averaged = np.mean(data.reshape(-1, n), axis=1)
            return np.where(averaged > 0.5, 1, 0)
        except:
            print('not divisible by ', n)

    def modulate(self, data):
        length = len(data)
        t = np.linspace(0, length, length)
        mod_high = np.multiply(np.sin(self.freq_high * t * 2 * np.pi), data)
        mod_low = np.multiply(np.sin(self.freq_low * t * 2 * np.pi), 1 - data)
        return mod_high + mod_low

    def demodulate(self, data, freq_high, freq_low):
        t = np.linspace(0, 1 / self.fs, self.rate)

        sin_high = np.sin(freq_high * t * 2 * np.pi)
        sin_low = np.sin(freq_low * t * 2 * np.pi)

        data_matrix = np.reshape(data, (len(data) // self.rate, self.rate))
        sol_high = np.dot(sin_high, np.transpose(data_matrix))
        sol_low = np.dot(sin_low, np.transpose(data_matrix))

        diff = sol_high - sol_low
        demodulated = np.abs(np.ceil(diff / self.rate))
        return demodulated

    def doubleDemodulate(self, data):
        part1 = self.demodulate(data, self.freq_high, self.freq_low)
        part2 = self.demodulate(data, self.f3, self.f4)
        return np.concatenate((part1, part2))

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

    def removeDoubleModPilots(self, singleDemod, originalData):
        pilot_1_converted = self.convertToOneMinusOne(self.pilot1.astype(np.float32))
        pilot_2_converted = self.convertToOneMinusOne(self.pilot2.astype(np.float32))

        offset_1 = self.calculateOffsetToTransmition(pilot_1_converted, self.convertToOneMinusOne(singleDemod)) - 50
        trunc_1 = singleDemod[offset_1 + len(self.pilot1):]
        offset_2 = self.calculateOffsetToTransmition(pilot_2_converted, self.convertToOneMinusOne(trunc_1)) - 50

        result = originalData[self.rate * (offset_1 + len(self.pilot1)):]
        return result[:self.rate * offset_2]

    def findOffsetToFirstChange(self, data):
        firstChange = self.modulate(self.repencode(np.array([1, 0])))
        return self.calculateOffsetToTransmition(firstChange, data)

    def bitsToBytes(self, bits):
        binaryBites = np.reshape(bits, ((len(bits) // 8), 8))
        dataAsBytes = np.packbits(binaryBites, axis=1).flatten().tobytes()
        return dataAsBytes

    def writeToFile(self, path, data):
        file = open(path, "wb")
        file.write(data)
        file.close()

    def recordAudio(self):
        seconds = 5 * 60
        myrecording = sd.rec(int(seconds * self.audioSampleRate), samplerate=self.audioSampleRate, channels=1)
        sd.wait()  # Wait until recording is finished

        file_name = 'recording.wav'
        recording = np.reshape(myrecording, myrecording.shape[0])
        scipy.io.wavfile.write(file_name, self.audioSampleRate, recording.astype(np.float32))
        return recording


    def test(self):
        input = self.readWav('recording.wav')
        #input = self.recordAudio()
        #print(self.demodulate(input))
        #input = np.concatenate((np.random.rand(1000), input, np.random.rand(1000)))

        offset = self.findOffsetToFirstChange(input)
        truncated = self.truncateToTauS(input, offset)
        #truncated = input
        decoded = self.demodulate(truncated, self.freq_high, self.freq_low)
        actual = self.removePilots(decoded)
        actual = self.repdecode(actual, 3)

        b = self.bitsToBytes(actual.astype(np.uint8))
        self.writeToFile('p.png', b)

        #expected = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
        #diff = expected - actual
        #error_sum = np.sum(np.abs(diff))

        audio = input * (2 ** 15 - 1) / np.max(np.abs(input))

        audio = audio.astype(np.int16)
        #play_onj = sa.play_buffer(audio, 1, 2, self.audioSampleRate)

        #play_onj.wait_done()

        print('actual: ', actual)
        #print('difference: ', diff)
        #print('error sum ', error_sum)

    def testDoubleDecode(self):
        input = self.readWav('test_double.wav')
        #input = self.recordAudio()
        truncated = self.truncateToTauS(input, self.findOffsetToFirstChange(input))
        singleDecoded = self.demodulate(truncated, self.freq_high, self.freq_low)
        noPilots = self.removeDoubleModPilots(singleDecoded, truncated)
        doubleDemod = self.doubleDemodulate(noPilots)
        actual = self.repdecode(doubleDemod, 3)

        print('actual: ', actual)



    def testDecode(self):
        a = self.repdecode(np.array([1,1,1,0,0,1,0,0,1,0,1,1]), 4)
        print(a)