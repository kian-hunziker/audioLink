from Sender import Sender
from Receiver import Receiver
import scipy.io.wavfile
import numpy as np
from Hamming import Hamming




sender = Sender()
sender.setTransmitionAmplitudes(0.46879888, 0.1121671319)
sender.sendCalibration()


receiver = Receiver()
print(receiver.getAvailableAudioDevices())
#receiver.setAudioInputDevice(4)
actual = receiver.calibrate(plot=True)





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