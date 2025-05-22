#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/31/周二 9:07
# @Author  : FangFano
# @File    : to96k.py
# @Software: PyCharm

import librosa
import soundfile as sf

# load original file
data, sample_rate = librosa.load('3_bandpass.wav')

# Set the target sampling rate
target_sample_rate = 96000

# Resampling audio data
data_resampled = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sample_rate)

# save to file
sf.write('4_96k.wav', data_resampled, target_sample_rate)