import numpy as np
import json
import torch
import scipy.io.wavfile
from scipy.fftpack import dct
from torch import nn
from torch.autograd import Variable

for i in range(0, 100):
    sample_rate, signal = scipy.io.wavfile.read("../../dataset/train/negative/" + str(i) + "/audio.wav")
    # signal = signal[0: int(6.5*sample_rate)]
    t = np.linspace(0, 30, num=len(signal))

    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis*signal[:-1])
    frame_size = 1
    frame_stride = 1
    frame_length, frame_step = frame_size*sample_rate, frame_stride*sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros(int(pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames*frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    ham = np.hamming(frame_length)
    # plt.plot(ham)
    # plt.show()

    frames *= ham

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  #(348, 257)

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate/2) / 700))

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)

    hz_points = (700 * (10**(mel_points / 2595) - 1))

    bin = np.floor((NFFT + 1)*hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m-1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m+1])

        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k-bin[m-1]) / (bin[m]-bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (num_ceps / 2) * np.sin(np.pi * n / num_ceps)
    mfcc *= lift

    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    mfcc = mfcc.reshape((mfcc.size, 1))
    np.save("../../dataset/train/negative/" + str(i) + "/audio.npy", mfcc)

for i in range(0, 100):
    sample_rate, signal = scipy.io.wavfile.read("../../dataset/train/positive/" + str(i) + "/audio.wav")
    # signal = signal[0: int(6.5*sample_rate)]
    t = np.linspace(0, 30, num=len(signal))

    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis*signal[:-1])
    frame_size = 1
    frame_stride = 1
    frame_length, frame_step = frame_size*sample_rate, frame_stride*sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros(int(pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames*frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    ham = np.hamming(frame_length)
    # plt.plot(ham)
    # plt.show()

    frames *= ham

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  #(348, 257)

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate/2) / 700))

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)

    hz_points = (700 * (10**(mel_points / 2595) - 1))

    bin = np.floor((NFFT + 1)*hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m-1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m+1])

        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k-bin[m-1]) / (bin[m]-bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (num_ceps / 2) * np.sin(np.pi * n / num_ceps)
    mfcc *= lift

    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    mfcc = mfcc.reshape((mfcc.size, 1))
    np.save("../../dataset/train/positive/" + str(i) + "/audio.npy", mfcc)
