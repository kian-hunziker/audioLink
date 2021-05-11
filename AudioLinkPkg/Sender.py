

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
        self.freq_high = 1/ 16
        self.freq_low = 1/ 20

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
        a = np.abs(np.ceil(diff / self.rate))
        return a

    def writeToWav(self, data):
        file_name = 'test_with_silence.wav'
        scipy.io.wavfile.write(file_name, self.audioSampleRate, data.astype(np.float32))

    def test(self):
        data = self.addPilots(self.repencode(self.getTestDataAsBits(), 3))
        encoded = self.repencode(data, self.rate)
        modulated = self.modulate(encoded)
        #self.writeToWav(np.concatenate((np.zeros(3*44100),modulated)))
        demodulated = self.demodulate(modulated)
        print(demodulated)
        self.playAudio(self.modulate(encoded))




