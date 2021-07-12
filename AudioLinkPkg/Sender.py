

import numpy as np
import simpleaudio as sa
import scipy.io
import scipy.io.wavfile



class Sender:

    def __init__(self):
        # sampling frequency
        self.fs = 1/160
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

    def playAudio(self, data):
        audio = data * (2 ** 15 - 1) / np.max(np.abs(data))

        audio = audio.astype(np.int16)
        play_onj = sa.play_buffer(audio, 1, 2, self.audioSampleRate)

        play_onj.wait_done()


    def getTestTone(self):
        frequency = 440
        seconds = 3
        t = np.linspace(0, seconds, seconds * self.fs, False)
        note = np.sin(frequency * t * 2 *np.pi)
        return note

    def getTestDataAsBits(self):
        #s = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1], dtype=np.uint8)
        s = np.array([1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,0,0,1,0,1,0,1,0,1,0], dtype=np.uint8)
        return s

    def repencode(self, data, n):
        encoded = np.repeat(data, n)
        return encoded

    def addPilots(self, data):
        return np.concatenate((self.pilot1, data, self.pilot2))

    def addModulatedPilots(self, data):
        modPilot1 = self.modulate(self.repencode(self.pilot1, self.rate))
        modPilot2 = self.modulate(self.repencode(self.pilot2, self.rate))
        return np.concatenate((modPilot1, data, modPilot2))

    def modulate(self, data):
        length = len(data)
        t = np.linspace(0, length, length)
        mod_high = np.multiply(np.sin(self.freq_high * t * 2 * np.pi), data)
        mod_low = np.multiply(np.sin(self.freq_low * t * 2 * np.pi), 1 - data)
        return mod_high + mod_low

    def doubleModulate(self, data):
        if not (len(data) % 2 == 0):
            print('we need padding or something of the sort')
            return
        length = len(data) // 2
        dataPart1 = data[0:length]
        dataPart2 = data[length:]

        t = np.linspace(0, length, length)
        mod_1 = np.multiply(np.sin(self.freq_high * t * 2 * np.pi), dataPart1)
        mod_2 = np.multiply(np.sin(self.freq_low * t * 2 * np.pi), 1 - dataPart1)
        mod_3 = np.multiply(np.sin(self.f3 * t * 2 * np.pi), dataPart2)
        mod_4 = np.multiply(np.sin(self.f4 * t * 2 * np.pi), 1 - dataPart2)

        return mod_1 + mod_2 + mod_3 + mod_4


    def demodulate(self, data, freq_high, freq_low):
        t = np.linspace(0, 1 / self.fs, self.rate)

        sin_high = np.sin(freq_high * t * 2 * np.pi)
        sin_low = np.sin(freq_low * t * 2 * np.pi)

        data_matrix = np.reshape(data, (len(data) // self.rate, self.rate))
        sol_high = np.dot(sin_high, np.transpose(data_matrix))
        sol_low = np.dot(sin_low, np.transpose(data_matrix))

        diff = sol_high - sol_low
        a = np.abs(np.ceil(diff / self.rate))
        return a

    def doubleDemodulate(self, data):
        part1 = self.demodulate(data, self.freq_high, self.freq_low)
        part2 = self.demodulate(data, self.f3, self.f4)
        return np.concatenate((part1, part2))

    def writeToWav(self, data):
        file_name = 'test_double.wav'
        scipy.io.wavfile.write(file_name, self.audioSampleRate, data.astype(np.float32))

    def readFromFile(self, path):
        file = open(path, "rb")
        data = file.read()
        file.close()
        return data

    def writeToFile(self, path, data):
        file = open(path, "wb")
        file.write(data)
        file.close()

    def test(self):
        data = self.addPilots(self.repencode(self.getTestDataAsBits(), 3))
        #dataBytes = self.readFromFile('penguin.png')
        #data = self.bytesToBits(dataBytes)
        encoded = self.repencode(data, self.rate)
        modulated = self.doubleModulate(encoded)
        self.writeToWav(np.concatenate((np.zeros(3*44100), modulated)))
        demodulated = self.doubleDemodulate(modulated)
        #demodulated = self.doubleDemodulate(modulated)
        print('data and pilots')
        print(demodulated)
        print('data only')
        print(self.getTestDataAsBits())
        #b = self.bitsToBytes(demodulated.astype(np.uint8))
        #self.writeToFile("pinguuuu.png", b)
        self.playAudio(self.modulate(encoded))

    def bytesToBits(self, data):
        dataAsInts = np.fromstring(data, dtype=np.uint8)
        bits = np.unpackbits(dataAsInts)
        return bits

    def bitsToBytes(self, bits):
        binaryBites = np.reshape(bits, ((len(bits) // 8), 8))
        dataAsBytes = np.packbits(binaryBites, axis=1).flatten().tobytes()
        return dataAsBytes

    def testConversion(self):
        data = self.readFromFile('penguin.png')
        bits = self.bytesToBits(data)
        res = self.bitsToBytes(bits)
        passed = data == res

    def testDoubleModulation(self):
        data = self.repencode(self.getTestDataAsBits(), 3)
        encoded = self.repencode(data, self.rate)
        modulatedData = self.doubleModulate(encoded)
        dataWithPilots = self.addModulatedPilots(modulatedData)
        self.writeToWav(dataWithPilots)
        self.playAudio(dataWithPilots)


