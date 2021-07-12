from Sender import Sender
from Receiver import Receiver
import scipy.io.wavfile

sender = Sender()
sender.testDoubleModulation()

receiver = Receiver()
receiver.testDoubleDecode()
