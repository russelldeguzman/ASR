import sounddevice as sd
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
from scipy.fftpack import dct
import wave
import sys
import pylab
from silence_detect import find_speech

def preemphasis_filter(s, alpha):
    filtered_sig = np.append(s[0], s[1:] - alpha * s[:-1])
    return filtered_sig

def window_sig(signal, fr_len, fr_step):
    fr_len = int(fr_len)
    fr_step = int(fr_step)
    nframes = int((len(signal) - fr_len)/ fr_step)

    #zero pad the signal so that we have enough to fill
    #the frames
    len_framed_sig = (nframes * fr_step) + fr_len

    padding = np.zeros(np.abs(len_framed_sig - len(signal)))
    signal = np.hstack((signal, padding))
    framed_sig  = np.ndarray((nframes, fr_len))

    for frame in range(len(framed_sig)):
        start = frame * fr_step
        stop = start + fr_len
        framed_sig[frame] = signal[start : stop]

    return framed_sig

#triangle filters
def triangle_filter(nfilters, mel_bin, fbanks_arr):
    #apply triangle
    for i in range(1, nfilters + 1):
        left = int(mel_bin[i - 1])
        center = int(mel_bin[i])
        right = int(mel_bin[i + 1])

        for j in range(left, center):
            fbanks_arr[i - 1, j] = (j - mel_bin[i - 1]) / (mel_bin[i] - mel_bin[i - 1])
        for j in range(center, right):
            fbanks_arr[i - 1, j] = (mel_bin[i + 1] - j) / (mel_bin[i + 1] - mel_bin[i])

    return fbanks_arr

#mel points
def get_mel_bins(fs, nfft, nfilters):
    low_mel = 0
    high_mel = (2595 * np.log10(1 + (fs / 2) / 700)) #hz to mel
    mel_points = np.linspace(low_mel, high_mel, nfilters + 2) #mel to hz
    freq_points = (700 * (10 ** (mel_points / 2595) - 1))
    mel_bins = np.floor((nfft + 1) * freq_points / fs)
    return mel_bins

#calculte the mel fitler bank
def mel_filter_bank(frame, nfilters, fs, fr_len):

    #get the mel freq bins
    mel_bin = get_mel_bins(fs, fr_len, nfilters)

    fbanks_arr = np.zeros((nfilters, int(fr_len)))

    #apply triange filter
    fbanks = triangle_filter(nfilters, mel_bin, fbanks_arr)

    fbanks = fbanks.T

    filter_banks = np.dot(frame, fbanks)

    return filter_banks

#take the mfcc
def get_mfcc(signal,fr_len,fr_step,fs, num_coef):
    mfcc = []
    #window signal
    windowed_sig = window_sig(signal,fr_len,fr_step)

    #silence processing
    windowed_sig = find_speech(windowed_sig,fs)

    #fft signal
    for window in range(0, len(windowed_sig)):
        #fft signal
        fft_sig = np.fft.fft(windowed_sig[window])
        # magnitude of signal
        fft_mag = np.absolute(fft_sig)
        # mel filter bank
        mel_bank = mel_filter_bank(fft_mag, 20, fs, fr_len)
        # log bank
        mel_bank = np.log10(mel_bank)
        #ifft (DCT)
        mfcc.append(dct(mel_bank))

    mfcc = np.asarray(mfcc)
    return mfcc[:, 0 : num_coef]

def compute_deltas(mfcc, window_size):
    result = np.zeros(((len(mfcc) - 2*window_size),(window_size * len(mfcc[0]))))
    deltas = np.zeros((len(mfcc),len(mfcc[0])))
    for j in range(0, len(mfcc[0])):
        for t in range(window_size,len(mfcc) - window_size):
            xtj = 0
            numerator = 0
            denominator = 0
            for m in range(-window_size, window_size + 1):
                numerator = numerator + (m*mfcc[t+m][j])
                denominator = denominator + (m**2)
            xtj = numerator / denominator
            deltas[t][j] = xtj
    result = np.concatenate((mfcc,deltas), axis = 1)
    return result.T;

def ret_mfcc(speech_sig, fs):
    FRAME_SIZE_MS = 25 #25ms
    FRAME_STEP_MS = 10 #ms
    NUM_CEPSTRUM = 13 # num of cepstral coefficients

    s = preemphasis_filter(speech_sig, 0.97)
    fr_len = (float(FRAME_SIZE_MS) / 1000) * fs
    fr_step = (float(FRAME_STEP_MS) / 1000) * fs

    mfcc = get_mfcc(s,fr_len,fr_step,fs, NUM_CEPSTRUM)
    ret = compute_deltas(mfcc, 2)
    return ret
