#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/31/周二 9:07
# @Author  : FangFano
# @File    : denoise.py
# @Software: PyCharm

from scipy.io import wavfile
import noisereduce as nr

# load original file
sample_rate, data = wavfile.read("1_single.wav")
# Executive noise suppression
reduced_noise = nr.reduce_noise(y=data, sr=sample_rate)

# save to file
wavfile.write("2_denoise.wav", sample_rate, reduced_noise)