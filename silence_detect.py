import numpy as np
import sounddevice as sd
import time

def get_sig_energy(signal, delta):
    ret = np.zeros(len(signal) - delta)
    for s in range(0,len(signal)-delta):
        nrg = 0
        for i in range(0, delta):
            nrg = nrg + signal.item(s+i)**2
        ret[s] = nrg
    return ret

def get_N1(signal,lt, ut):
    uindex = np.amin(np.argwhere(signal > ut))
    lindex = np.amin(np.argwhere(signal > lt)) #TODO these need work
    new_index_needed = False
    # print lt
    # print ut
    # print lindex
    # print uindex
    result = lindex
    for l in range(lindex, uindex):
        if signal[l] < lt:
            new_index_needed = True
        if signal[l] > lt and new_index_needed:
            lindex = l
            new_index_needed = False
    return result

def search_N1_prime(signal, n1, zero_cross_thresh):
    step = 1
    zero_crossings = np.argwhere(signal[(n1-step):n1] > zero_cross_thresh)
    if len(zero_crossings) >= 3:
        return zero_crossings[0]
    else:
        return n1

def search_N2_prime(signal, n2, zero_cross_thresh):
    step = 1
    zero_crossings = np.argwhere(signal[n2:(n2+step)] > zero_cross_thresh)
    if len(zero_crossings) >= 3:
        return zero_crossings[-1]
    else:
        return n2

def get_zx_arr(signal, Fs, delta):
    ret = np.zeros(len(signal)-delta)
    for n in range(1,len(signal)-delta):
        zx = 0
        for i in range(0, delta):
            zx = zx + np.absolute(np.sign(signal.item(n+i)) - np.sign(signal.item(n+i-1)))/2
        ret[n] = zx / delta
    return ret

def get_N2(signal,lt,ut):
    lindex = np.amax(np.argwhere(signal > lt))
    return lindex

def rabiner_sambur(signal, Fs):
    #calc delta
    delta = 1 #25ms
    trim = 10 # trim the window so we capture the main signal
    signal = signal[trim: len(signal)]
    #compute energy of signal
    sig_nrg = get_sig_energy(signal, delta)
    # plt.plot(sig_nrg)
    # plt.show()

    #search for the actual point 25 ms before
    zx_arr = get_zx_arr(signal, Fs, delta)
    # plt.plot(zx_arr)
    # plt.show()

    #compute min and max of energy
    nrg_min = np.amin(sig_nrg)
    nrg_max = np.amax(sig_nrg)

    #set energy thresholds
    lower_nrg_thresh = 0.03*(nrg_max-nrg_min)+nrg_min
    upper_nrg_thresh = 5 * lower_nrg_thresh
    #set zero crossing threshold
    zero_crossing_thresh = min(0.25, np.mean(zx_arr[0: 100])) # 250ms

    #find first speech point
    n1 = get_N1(sig_nrg, lower_nrg_thresh, upper_nrg_thresh)

    #search for the actual point 25 ms before
    n1_p = search_N1_prime(zx_arr, n1, zero_crossing_thresh)

    #find n2
    n2 = get_N2(sig_nrg, lower_nrg_thresh, upper_nrg_thresh)

    #TODO: Check this
    n2_p = search_N2_prime(zx_arr, n2, zero_crossing_thresh)

    return (n1_p,n2_p)

def find_speech(speech_signal, Fs):
    n1,n2 = rabiner_sambur(speech_signal, Fs)
    return speech_signal[n1:n2]
