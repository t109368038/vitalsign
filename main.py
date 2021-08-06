import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

fft_cfg = {"fftpoints":128}

ncsum_cfg = {"Nsum": 4,
            "cnt": 0,
            "sum_abs_fft_sumchirp": np.zeros([int(fft_cfg["fftpoints"]),1])
             }

Cfar_cfg = {"MinPeakProminence": 5,
            "MinPeakDistance": 3,
            "guardLen": 4,
            "noiseWin": 8,
            "threshold1D": 10,
            "cfarmode": 4,
            "startIdx": 5,
            "done": 0}

def fft_process(data, fft_points):
    window = np.hanning(64)
    fft1d_in = data * window
    fft1d_out = np.fft.fft(fft1d_in, n=fft_points, axis=1)
    return fft1d_out

def whole_detect_process(fft_sumofchirp):
    abs_fft_sumchirp = np.abs(fft_sumofchirp)

    non_coherentsum(abs_fft_sumchirp, ncsum_cfg)

    if (ncsum_cfg["cnt"] % ncsum_cfg["Nsum"]) == 0 :
        print(ncsum_cfg["sum_abs_fft_sumchirp"].size)
        DB_sum_abs_fft_sumchirp = 10 * np.log10(ncsum_cfg["sum_abs_fft_sumchirp"])
        plot_data(DB_sum_abs_fft_sumchirp)
        print(ncsum_cfg["cnt"])
        # find_peaks(DB_sum_abs_fft_sumchirp)

def non_coherentsum(abs_fft_sumchirp, ncsum_cfg):

    if ncsum_cfg["cnt"] > ncsum_cfg["Nsum"]:
        ncsum_cfg["sum_abs_fft_sumchirp"] = abs_fft_sumchirp

    elif ncsum_cfg["cnt"] == ncsum_cfg["Nsum"]:
        ncsum_cfg["cnt"] = 1
        ncsum_cfg["sum_abs_fft_sumchirp"] = abs_fft_sumchirp

    elif ncsum_cfg["cnt"] < ncsum_cfg["Nsum"]:
        ncsum_cfg["sum_abs_fft_sumchirp"] += abs_fft_sumchirp
        ncsum_cfg["cnt"] += 1

def plot_heatmap(data):
    ax = plt.imshow(np.abs(data))
    plt.show()

def plot_data(data):
    ax = plt.plot(data)
    plt.show()

if __name__ == '__main__':
    "==============================="
    "======= config setting ========"
    "==============================="

    radar_data = np.load('C:/python_proj/thmouse_training_data/right/time2/raw.npy', allow_pickle=True)
    data = np.reshape(radar_data, [-1, 4])
    data = data[:, 0:2:] + 1j * data[:, 2::]
    rawData = np.reshape(data, [-1, 16, 12, 64])
    for i in range(len(rawData)):
        tmp = rawData[0,:,0,:]
        print(tmp.shape)

        fft1d_out = fft_process(tmp, fft_cfg["fftpoints"])
        print(np.shape(fft1d_out))

        fft_tmp_chirp_sum = np.sum(fft1d_out, axis=0)
        fft_tmp_chirp_sum = fft_tmp_chirp_sum.reshape([int(fft_cfg["fftpoints"]),1])

        whole_detect_process(fft_tmp_chirp_sum)


    print("sstop")
    plot_data(fft_tmp_chirp_sum)