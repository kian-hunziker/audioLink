

import numpy as np
import simpleaudio as sa



class Sender:

    def __init__(self):
        self.fs = 44100

    def playAudio(self, data):
        audio = data * (2 ** 15 - 1) / np.max(np.abs(data))

        audio = audio.astype(np.int16)
        play_onj = sa.play_buffer(audio, 1, 2, self.fs)

        play_onj.wait_done()


    def getTestTone(self):
        frequency = 440
        seconds = 3

        t = np.linspace(0, seconds, seconds * self.fs, False)

        note = np.sin(frequency * t * 2 *np.pi)

        return note




