import sounddevice as sd
import scipy.io.wavfile
from scipy import signal
import numpy as np
import simpleaudio as sa
import sounddevice as sd
from Sender import Sender
from scipy.io.wavfile import write
from Hamming import Hamming

class Receiver:
    def __init__(self, tauS=160, tau0=16, tau1=40, sample_rate=44100):
        '''
        :param tauS: determines how many samples are used to modulate one bit
        tauS must be multiple of both tau0 and tau1
        :param tau0: determines the frequency of the high modulation note
        :param tau1: determines the frequency of the low modulation
        :param sample_rate: determines how many audio samples are used per second
        '''

        # sanity check to see if tauS is indeed a multiple of tau0 and tau1
        checkTau0 = tauS // tau0
        checkTau1 = tauS // tau1

        if not (checkTau0 * tau0 == tauS and checkTau1 * tau1 == tauS):
            print('tauS must be multiple of both tau0 and tau1')
            return

        self.fs = 1/tauS
        self.rate = tauS
        self.freq_high = 1 / tau0
        self.freq_low = 1 / tau1

        self.weight_high = 1
        self.weight_low = 1

        # could be used for double modulation. Not in use as of now
        self.f3 = 1 / 40
        self.f4 = 1 / 16

        self.audioSampleRate = sample_rate
        self.audioDeviceId = 0

        self.hamming = Hamming()


    def getTestBits(self, repetitions):
        return np.tile(np.array([1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,0,0,1,0,1,0,1,0,1,0]), repetitions)

    def readWav(self, file_name) -> np.ndarray:
        rate, data = scipy.io.wavfile.read(file_name)

        if data.dtype == np.int16:
            return data.astype(np.float32, order='C') / 32768.0
        return data

    def repencode(self, data, n):
        encoded = np.repeat(data, n)
        return encoded

    def repdecode(data, n):
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

    def demodulate(self, data, freq_high, freq_low):
        t = np.linspace(0, 1 / self.fs, self.rate)

        sin_high = np.sin(freq_high * t * 2 * np.pi)
        sin_low = np.sin(freq_low * t * 2 * np.pi)

        data_matrix = np.reshape(data, (len(data) // self.rate, self.rate))
        sol_high = np.abs(np.dot(sin_high, np.transpose(data_matrix)))
        sol_low = np.abs(np.dot(sin_low, np.transpose(data_matrix)))

        diff = sol_high - sol_low
        demodulated = np.abs(np.ceil(diff / self.rate))
        return demodulated

    def doubleDemodulate(self, data):
        part1 = self.demodulate(data, self.freq_high, self.freq_low)
        part2 = self.demodulate(data, self.f3, self.f4)
        return np.concatenate((part1, part2))

    def calculateOffsetToTransmition(self, zeroOne, data):
        testCorr = signal.correlate(data, zeroOne, mode="same")

        # TODO improve this offset calculation
        indices = np.where(testCorr > np.max(testCorr) - 2)
        if len(indices) > 0 and len(indices[0] > 0):
            return indices[0][0]
        else:
            return indices

    def truncateToTauS(self, data, offset):
        truncated_start = data[(offset % self.rate):]
        res = truncated_start[:len(truncated_start) - (len(truncated_start) % self.rate)]
        return res

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
        firstChange = self.modulate(self.repencode(np.array([1, 0]), self.rate))
        return self.calculateOffsetToTransmition(firstChange, data)

    def bitsToBytes(self, bits):
        binaryBites = np.reshape(bits, ((len(bits) // 8), 8))
        dataAsBytes = np.packbits(binaryBites, axis=1).flatten().tobytes()
        return dataAsBytes

    def writeToFile(self, path, data):
        file = open(path, "wb")
        file.write(data)
        file.close()

    def recordAudio(self, duration, save_recording=False, recording_name=None):
        seconds = duration
        myrecording = sd.rec(int(seconds * self.audioSampleRate), samplerate=self.audioSampleRate, channels=1)
        sd.wait()  # Wait until recording is finished

        if save_recording:
            file_name = recording_name
            if not recording_name.endswith('.wav'):
                file_name = recording_name + '.wav'
            recording = np.reshape(myrecording, myrecording.shape[0])
            scipy.io.wavfile.write(file_name, self.audioSampleRate, recording.astype(np.float32))

        return recording

    def getAvailableAudioDevices(self):
        return sd.query_devices(device=None, kind=None)

    def setAudioInputDevice(self, device_id):
        self.audioDeviceId = device_id
        sd.default.device = device_id

    def gateInput(self, data):
        thresh = 2 * np.max(data[:self.audioSampleRate//2])
        return np.where(np.abs(data) < thresh, 0, data)

    def test(self, rec_duration, testBitRepetitions, encodeRepetitions, hamming):
        expected = self.getTestBits(testBitRepetitions)

        if hamming:
            actual = self.receiveHammingEncoded(rec_duration, repetitions=encodeRepetitions, bits=True,
                                                save_file=True, recording_name='lastTransmission.wav')
        else:
            actual = self.receiveRepencoded(rec_duration, repetitions=encodeRepetitions, bits=True,
                                                save_file=True, recording_name='lastTransmission.wav')

        print('actual: ', actual)
        print('length of actual:', len(actual))

        diff = expected - actual[:len(expected)]
        error_sum = np.sum(np.abs(diff))

        print('error sum ', error_sum)
        print('error weight', np.sum(diff))
        print('error percentage', error_sum / len(expected) * 100)

    def receiveRepencoded(self, duration, repetitions=5, bits=False, from_file=False, file_path=None,
                          save_file=False, recording_name=None):
        data_in = None
        if from_file:
            data_in = self.readWav(file_path)
        else:
            data_in = self.recordAudio(duration, save_file, recording_name)

        offset = self.findOffsetToFirstChange(data_in)

        if offset > self.audioSampleRate // 2 + self.rate // 2:
            data_in = self.gateInput(data_in)

        truncated = self.truncateToTauS(data_in, offset)
        demodulated = self.demodulate(truncated, self.freq_high, self.freq_low)
        no_pilots = self.removePilots(demodulated)
        decoded = self.repdecode(no_pilots, repetitions)

        if bits:
            return decoded
        else:
            try:
                data_as_bytes = self.bitsToBytes(decoded)
                return data_as_bytes
            except:
                print('could not convert bits to bytes. \nData might not be divisible by eight')

    def receiveHammingEncoded(self, duration, repetitions=5, bits=False, from_file=False, file_path=None,
                          save_file=False, recording_name=None):
        data_in = None
        if from_file:
            data_in = self.readWav(file_path)
        else:
            data_in = self.recordAudio(duration, save_file, recording_name)

        offset = self.findOffsetToFirstChange(data_in)

        if offset > self.audioSampleRate // 2 + self.rate // 2:
            data_in = self.gateInput(data_in)

        truncated = self.truncateToTauS(data_in, offset)
        demodulated = self.demodulate(truncated, self.freq_high, self.freq_low)
        no_pilots = self.removePilots(demodulated)
        rep_decoded = self.repdecode(no_pilots, repetitions)
        decoded = self.hamming.decodeAndCorrectStream(rep_decoded)

        if bits:
            return decoded
        else:
            try:
                data_as_bytes = self.bitsToBytes(decoded)
                return data_as_bytes
            except:
                print('could not convert bits to bytes. \nData might not be divisible by eight')


    def testDoubleDecode(self):
        #input = self.readWav('test_double.wav')
        input = self.recordAudio()
        truncated = self.truncateToTauS(input, self.findOffsetToFirstChange(input))
        singleDecoded = self.demodulate(truncated, self.freq_high, self.freq_low)
        noPilots = self.removeDoubleModPilots(singleDecoded, truncated)
        doubleDemod = self.doubleDemodulate(noPilots)
        actual = self.repdecode(doubleDemod, 3)

        print('actual: ', actual)

    def testDecode(self):
        a = self.repdecode(np.array([1,1,1,0,0,1,0,0,1,0,1,1]), 4)
        print(a)