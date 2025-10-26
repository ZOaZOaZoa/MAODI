import librosa.core as core
import librosa.feature as ftr
import librosa.util as ut
import matplotlib.pyplot as plt
import dataprocessing as dp
import soundfile as sf
import sys
import numpy as np

# dictionary = ['Вперед', 'Назад', 'Налево', 'Направо', 'Стоп', 'Разворот', 'Вверх', 'Вниз', 'Увеличить', 'Уменьшить', 'Сканировать', 'Осмотреться']
dictionary = ['Вперед']
# dictionary = ['Вперед', 'Назад', 'Налево', 'Направо', 'Стоп', 'Разворот']
path = "C:\\work\\PycharmProjects\\voice\\venv\\files\\"
frame_time = 0.02
frame_shift = 0.5

print(len(dictionary))

# for w_ind in range(len(dictionary)):
#     filename = dictionary[w_ind]
#     fname = path + filename + ".wav"
#     newfname = path + filename + '_edited' +  ".wav"
#     y, sr = sf.read(file=fname, dtype='int16')
#
#     # dp.resave(fname, 3500, 877475, newfname)
#
#
#     frameWidth = int(frame_time * sr)
#     frameShift = int(frame_shift * frameWidth)
#     df = frameWidth - frameShift
#     frameCount = int(y.size / df) - 1
#
#     # enData = dp.dataToEnergy(y, sr=sr, frame_time=frame_time, frame_shift=frame_shift)
#
#     # plt.figure()
#     # plt.plot(enData)
#
#     vadData, tickData = dp.VAD(y, sr=sr, frame_time=frame_time, frame_shift=frame_shift, noise_frame_end=20,eTh=0)
#     # vadData, tickData = dp.VAD(y, sr=sr, frame_time=frame_time, frame_shift=frame_shift, noise_frame_end=0,eTh=8000)
#
#     maxY = np.max(y)
#     vadData2 = [0] * y.size
#     for j in range(frameCount):
#         if vadData[j] == 1:
#            for i in range(frameWidth):
#                vadData2[j * df + i] = maxY
#         else:
#             for i in range(frameWidth):
#                 vadData2[j * df + i] = 0
#
#     plt.figure()
#     plt.plot(vadData)
#
#     plt.figure()
#     plt.plot(y)
#     plt.plot(vadData2)
#
#     dt = []
#     for j in range(int(len(tickData) / 2)-1):
#         dt.append(tickData[j * 2 + 2] - tickData[j * 2 + 1])
#
#     hist,bin_edges = np.histogram(dt)
#     # bins = plt.hist(dt)
#     # print(bins)
#     print(bin_edges)
#
#     min_pause = (int(bin_edges[1] / df) + 1) * df
#     print(min_pause)
#     min_pause = 10 * frameWidth
#
#     tickData2 = []
#     tickData2.append(tickData[0])
#     for j in range(int(len(tickData) / 2) - 1):
#         pause = tickData[j * 2 + 2] - tickData[j * 2 + 1]
#         if pause > min_pause:
#             tickData2.append(tickData[2 * j + 1])
#             tickData2.append(tickData[2 * j + 2])
#
#     tickData2.append(tickData[len(tickData) - 1])
#
#     vadData3 = [0] * y.size
#     for j in range(int(len(tickData2) / 2)):
#         for i in range(tickData2[2 * j], tickData2[2 * j + 1]):
#                 vadData3[i] = maxY / 2
#     plt.plot(vadData3)
#
#     plt.figure()
#     bins = plt.hist(dt)
#
# plt.show()
#
# sys.exit(-1)



for w_ind in range(len(dictionary)):
    filename = dictionary[w_ind]

    fname = path + filename + ".wav"

    y, sr = sf.read(file=fname,dtype='int16')
    # y, sr = core.load(fname, sr = 22050, mono = True)

    frameWidth = int(frame_time * sr)
    # print(frameWidth)

    frameShift = int(frame_shift * frameWidth)
    # print(frameShift)
    df = frameWidth - frameShift

    frameCount = int(y.size / df) - 1

    # print(frameCount)


    # x_frames = ut.frame(y, frame_length=frameWidth, hop_length=frameShift, axis=0)

    # print (len(x_frames))

    # enData = dp.dataToEnergy(y, sr=sr, frame_time=frame_time, frame_shift=frame_shift)

    vadData, tickData = dp.VAD(y, sr=sr, frame_time=frame_time, frame_shift=frame_shift, noise_frame_end=0,eTh=12580)

    dt = []
    for j in range(int(len(tickData) / 2) - 1):
        dt.append(tickData[j * 2 + 2] - tickData[j * 2 + 1])

    min_pause = 10 * frameWidth

    tickData2 = []
    tickData2.append(tickData[0])
    for j in range(int(len(tickData) / 2) - 1):
        pause = tickData[j * 2 + 2] - tickData[j * 2 + 1]
        if pause > min_pause:
            tickData2.append(tickData[2 * j + 1])
            tickData2.append(tickData[2 * j + 2])

    tickData2.append(tickData[len(tickData) - 1])



    # for j in range(int(len(tickData2) / 2)):
    #     out = y[tickData2[j * 2] - 10 * frameWidth:tickData2[j * 2 + 1] + 10 * frameWidth]
    #     sf.write(file=path + "samples\\sample1\\all\\" + filename + str(j + 1) + '.wav', data=out, samplerate=sr, subtype='PCM_16')

# print(tickData)
#
# fig1 = plt.figure()
# plt.plot(enData)

fig2 = plt.figure()
plt.plot(vadData)
fig1 = plt.figure()
plt.hist(y, bins=24, density=True)

fig2 = plt.figure()
plt.plot(y)

plt.show()
