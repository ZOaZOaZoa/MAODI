import wave
import pyaudio as py
import keyboard
from array import array
import os
import dataprocessing as dp
import numpy as np
import soundfile as sf

def main():
    path = os.getcwd() + '\\'
    command = 'Осмотреться'
    dir = path + command + '\\'
    os.makedirs(dir, exist_ok=True)

    fname = dir + f"{command}.wav"

    CHUNK = 1024
    FORMAT = py.paInt16
    CHANNELS = 1
    RATE = 22050

    FRAME_TIME = 20E-3
    FRAME_SHIFT = 0.5

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
                print("Началась запись")

            data = stream.read(CHUNK, exception_on_overflow=False)
            data_all.extend(array('h', data))
    print('Вышли из цикла')

    stream.stop_stream()
    stream.close()
    p.terminate()
    print('Запись остановлена')

    wav = wave.open(fname, "wb")
    wav.setnchannels(CHANNELS)
    wav.setframerate(RATE)
    wav.setsampwidth(p.get_sample_size(FORMAT))
    wav.writeframes(data_all)
    wav.close()
    print(f"Запись сохранена в {fname}")

    E_noise_max = dp.getEMax(fname, seconds_from_start=1, frame_time=FRAME_TIME, frame_shift=FRAME_SHIFT)
    data, samplerate = sf.read(fname)
    _, ticks = dp.VAD(data, samplerate, FRAME_TIME, FRAME_SHIFT, noise_frame_end=0, eTh= E_noise_max)

    counter = 1
    for i in range(int(len(ticks) / 2)):
        left_tick = ticks[2*i]
        right_tick = ticks[2*i + 1]
        if right_tick - left_tick < 5000:
            continue
        
        dp.resave(fname, left_tick, right_tick, fname.replace('.wav', f'{counter}.wav'))
        counter += 1
    

if __name__ == '__main__':
    main()