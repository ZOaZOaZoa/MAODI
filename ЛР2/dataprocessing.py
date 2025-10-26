import numpy as np
import soundfile as sf

def acf(x):
    n = len(x)
    print(n)
    if n > 0:
        res = [0] * (2 * n - 1)
        mx = 0
        for j in range(n):
            mx = mx + x[j]
        mx = mx / n

        for j in range(n):
            x[j] = (x[j] - mx)
            res[n - 1] = res[n - 1] + x[j] * x[j]
        res[n - 1] = res[n - 1] / (n)
        print(res[n - 1])

        for k in range(1, n):
            # res[k + n - 1] = 0
            nmax = n - k
            for j in range(nmax):
                res[n + k - 1] = res[n + k - 1] + x[j] * x[j + k]
            res[n + k - 1] = res[n + k - 1] / (n - k)
            res[n - 1 - k] = res[n + k - 1]
    else:
        res = []

    # res2 =  [0] * (2 * n - 1)
    # for j in range(2*n - 1):
    #     res2[j] = res[j] / res[n - 1]
    print(x[0])
    print(x[n-1])
    return res

def acf2(x):
    n = len(x)
    mx = 0
    for j in range(n):
        mx = mx + x[j]
    mx = mx / n

    for j in range(n):
        x[j] = (x[j] - mx)

    y = [0] * 2 * n
    for j in range(n):
        y[j] = x[j]

    for j in range(n, n):
        y[j] = 0

    n = len(y)
    a = np.fft.rfft(y, n)
    print(a)
    b = [0] * len(a)
    for j in range(len(a)):
        b[j] = a[j].real * a[j].real + a[j].imag * a[j].imag
    c = np.fft.ifft(b)
    c2 = np.fft.fftshift(c)
    n = len(x)
    res = [0] * n
    for j in range(n):
        res[j] = c2[j].real


    return res

def dataToEnergy(data, sr, frame_time, frame_shift):
    frameWidth = int(frame_time * sr)
    frameShift = int(frame_shift * frameWidth)

    frameCount = int(data.size / (frameWidth - frameShift)) - 1
    print(f'{frameWidth=}')
    print(f'{frame_shift=}')
    print(f'{frameCount=}')

    E = [0.0] * frameCount
    sh = 0
    df = frameWidth - frameShift
    for i in range(frameCount):
        En = 0
        print('A')
        for j in range(frameWidth):
            sn = float(data[j + sh] ** 2)
            En = En + sn

        E[i] = En / frameWidth
        sh = sh + df

    return E


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


def noise_reduction():

    return None