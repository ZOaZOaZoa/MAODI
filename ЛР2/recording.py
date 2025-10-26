import wave
import pyaudio as py
import keyboard
import sys
from array import array

path = "C:\\!МЭИ\\Мага\\MAODI\\ЛР2\\"
fname = path + "test_recording_new1.wav"


CHUNK = 1024
FORMAT = py.paInt16
CHANNELS = 1
RATE = 22050
RECORD_SECONDS = 5

frames = []
data_all = array('h')

p = py.PyAudio()


def callback(in_data, frame_count, time_info, status):
    data_chunk = array('h', in_data)
    data_all.extend(data_chunk)
    return(in_data, py.paContinue)

def key_press(e):
    if (e.name == 'enter'):
        stream.stop_stream()
        stream.close()

        keyboard.unhook_all()

        print('открвыю')
        wav = wave.open(fname, "wb")
        wav.setnchannels(CHANNELS)
        wav.setframerate(RATE)
        wav.setsampwidth(p.get_sample_size(FORMAT))
        wav.writeframes(data_all)
        wav.close
        print('закрываю')
        p.terminate()
        print('терминирую')
        exit()
        print('выход')


stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, start=False, stream_callback=callback)

stream.start_stream()


keyboard.hook(key_press)
keyboard.wait()

sys.exit(-1)




