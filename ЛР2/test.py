import wave
import pyaudio as py
import keyboard
from array import array
import os
import sys

path = os.getcwd() + '\\'
command = 'Вперёд'
dir = path + command + '\\'
os.makedirs(dir, exist_ok=True)

fname = dir + f"{command}.wav"

CHUNK = 1024
FORMAT = py.paInt16
CHANNELS = 1
RATE = 22050

p = py.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

data_all = array('h')

recording = True

def key_press(e):
    global recording
    if e.name == 'enter':
        recording = False
        keyboard.unhook_all()

keyboard.hook(key_press)

while recording:
    if stream.is_active():
        data = stream.read(CHUNK, exception_on_overflow=False)
        data_all.extend(array('h', data))

stream.stop_stream()
stream.close()
p.terminate()

wav = wave.open(fname, "wb")
wav.setnchannels(CHANNELS)
wav.setframerate(RATE)
wav.setsampwidth(p.get_sample_size(FORMAT))
wav.writeframes(data_all)
wav.close()

