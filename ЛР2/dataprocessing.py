import numpy as np
import soundfile as sf
import librosa.feature as ft

def dataToEnergy(data, sr, frame_time, frame_shift, do_print = True):
    frameWidth = int(frame_time * sr)
    frameShift = int(frame_shift * frameWidth)

    frameCount = int(data.size / (frameWidth - frameShift)) - 1
    if do_print:
        print(f'{frameWidth=}')
        print(f'{frame_shift=}')
        print(f'{frameCount=}')

    E = [0.0] * frameCount
    sh = 0
    df = frameWidth - frameShift
    for i in range(frameCount):
        En = 0
        for j in range(frameWidth):
            sn = float(data[j + sh] ** 2)
            En = En + sn

        E[i] = En / frameWidth
        sh = sh + df

    return E

def getEMax(file_name, seconds_from_start, frame_time, frame_shift):
    data, samplerate = sf.read(file_name)

    noise_indices = int(seconds_from_start * samplerate)
    noise_data = data[:noise_indices]

    E = dataToEnergy(noise_data, samplerate, frame_time, frame_shift, do_print=False)
    return np.max(E)

def VAD(data, sr, frame_time, frame_shift, noise_frame_end, eTh):
    frameWidth = int(frame_time * sr)
    frameShift = int(frame_shift * frameWidth)

    frameCount = int(data.size / (frameWidth - frameShift)) - 1

    maxY = np.max(data)
    df = frameWidth - frameShift
    E = [0] * frameCount
    sh = 0
    for i in range(frameCount):
        En = 0

        for j in range(frameWidth):
            sn = float(data[j + sh] ** 2)
            En = En + sn

        E[i] = En / frameWidth
        sh = sh + df

    if noise_frame_end > 0:
        ePorog = 0
        for j in range(0, noise_frame_end):
            if ePorog < E[j]:
                ePorog = E[j]

        print(ePorog)
        ePorog = ePorog * 5
    else:
        ePorog = eTh

    st_jcounter = False
    dj_len = 0
    tickData = []
    vadData = [0] * frameCount
    for j in range(frameCount):

        if E[j] <= ePorog:
            vadData[j] = 0
        else:
            vadData[j] = 1

        if (j > 0):
            if (vadData[j-1] < vadData[j]):
                st_jcounter = False
                if (j - dj_len >= 0):
                    tickData.append((j - dj_len) * df)
                else:
                    tickData.append(0 * df)
            elif ( vadData[j-1] > vadData[j]):
                st_jcounter = True
                dj = 0
                tickData.append(j * df)
            else:
                if st_jcounter:
                    dj+=1
                    if dj == dj_len:
                        tickData.append(j * df)
                        st_jcounter = False
                        dj = 0
        # if (j == frameCount - 1) and st_jcounter and (dj > 0):
        #     tickData.append(j * df)



    return vadData, tickData

def resave(fname, start, end, newfname):
    y, sr = sf.read(file=fname, dtype='int16')
    out = y[start:end]
    sf.write(file=newfname, data=out, samplerate=sr,subtype='PCM_16')

def resave_activity_fragments(fname, frame_time, frame_shift):
    E_noise_max = getEMax(fname, seconds_from_start=1, frame_time=frame_time, frame_shift=frame_shift)
    data, samplerate = sf.read(fname)
    _, ticks = VAD(data, samplerate, frame_time, frame_shift, noise_frame_end=0, eTh= E_noise_max)

    counter = 1
    for i in range(int(len(ticks) / 2)):
        left_tick = ticks[2*i]
        right_tick = ticks[2*i + 1]
        if right_tick - left_tick < 5000:
            continue
        
        resave(fname, left_tick, right_tick, fname.replace('.wav', f'{counter}.wav'))
        counter += 1
