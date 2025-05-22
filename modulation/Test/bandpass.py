#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/31/周二 9:07
# @Author  : FangFano
# @File    : bandpass.py
# @Software: PyCharm

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# Chose bandpass parameter
low_cut = 47.0
high_cut = 5000.0
order = 1

# Load .wav file
sample_rate, data = wavfile.read('2_denoise.wav')

# bandpass filter function
def butter_bandpass(low, high, fs, order=5):
    # Normalized to the nyquist frequency, i.e. half the sampling rate.
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Set bandpass parameter
b, a = butter_bandpass(low_cut, high_cut, sample_rate, order=order)

# Apply the bandpass filter
filtered_data = filtfilt(b, a, data)

# Save the processed audio file
wavfile.write('3_bandpass.wav', sample_rate, filtered_data.astype(np.int16))
