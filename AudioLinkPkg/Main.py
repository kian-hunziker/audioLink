from Sender import Sender
from Receiver import Receiver
import scipy.io.wavfile
import numpy as np
from Hamming import Hamming




sender = Sender(tau0=20, tau1=80)
sender.setTransmitionAmplitudes(0.54213, 0.26384)
#sender.sendCalibration()
testbits = sender.getTestDataAsBits(50)
data = sender.readFromFile('pacman2.bmp')
#sender.sendDataRepencoded(data, 3)
#sender.sendDataHamming(testbits, repetitions=3, bits=True)


receiver = Receiver(tau0=20, tau1=80)
#print(receiver.getAvailableAudioDevices())
#receiver.setAudioInputDevice(4)
#actual = receiver.calibrate(plot=True)
actual = receiver.receiveRepencoded(10, 3, plot=True, from_file=True, file_path='testFiles/pacmanAudio.wav')
print(actual)
print('sum', np.sum(np.abs(testbits - actual)))




def testHamming():
    sender = Sender()
    receiver = Receiver()
    hamming = Hamming()

    testBits = sender.getTestDataAsBits(5)
    hammEncoded = hamming.encodeBitStream(testBits)
    rependcoded = sender.repencode(hammEncoded, 3)
    withPilots = sender.addPilots(rependcoded)

    noPilots = receiver.removePilots(withPilots)
    repdec = receiver.repdecode(noPilots, 3)
    actual = hamming.decodeAndCorrectStream(repdec)

    errorSum = np.sum(np.abs(sender.getTestDataAsBits(5) - actual))
    print('errorSum hamming', errorSum)