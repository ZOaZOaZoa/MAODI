import wave
import pyaudio as py
import keyboard
from array import array
import os
import dataprocessing as dp
import numpy as np
import soundfile as sf

def record_voice(file_to_save, descriptive = True):
    CHUNK = 1024
    FORMAT = py.paInt16
    CHANNELS = 1
    RATE = 22050

    p = py.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    data_all = array('h')

    recording = True

    def key_press(e):
        nonlocal recording
        if e.name == 'enter':
            recording = False
            keyboard.unhook_all()

    keyboard.hook(key_press)

    start_prompted = False
    while recording:
        if stream.is_active():
            if not start_prompted:
                start_prompted = True
                if descriptive:
                    print("Началась запись")

            data = stream.read(CHUNK, exception_on_overflow=False)
            data_all.extend(array('h', data))
    
    if descriptive:
        print('Вышли из цикла')

    stream.stop_stream()
    stream.close()
    p.terminate()
    if descriptive:
        print('Запись остановлена')

    wav = wave.open(file_to_save, "wb")
    wav.setnchannels(CHANNELS)
    wav.setframerate(RATE)
    wav.setsampwidth(p.get_sample_size(FORMAT))
    wav.writeframes(data_all)
    wav.close()
    if descriptive:
        print(f"Запись сохранена в {file_to_save}")

def main():
    FRAME_TIME = 20E-3
    FRAME_SHIFT = 0.5

    path = os.getcwd() + '\\'
    command = 'тест'
    dir = path + command + '\\'
    os.makedirs(dir, exist_ok=True)

    fname = dir + f"{command}.wav"
    record_voice(fname)
    dp.resave_activity_fragments(fname, FRAME_TIME, FRAME_SHIFT)
    

if __name__ == '__main__':
    main()