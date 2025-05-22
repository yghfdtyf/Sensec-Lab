#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/30/周一 16:23
# @Author  : FangFano
# @File    : am.py
# @Software: PyCharm

import numpy as np
import wave
import struct
import math
import scipy.io.wavfile

def main():
    # Import the original audio file.
    # Sample rate must be 96khz!
    sample_rate, origin_data = scipy.io.wavfile.read("4_96k.wav")
    print("sample rate: ", sample_rate)
    if sample_rate != 96000:
        print("Sample rate must be 96khz!")
        return
    data_len = len(origin_data)
    print("data length: ", data_len)
    # Converts raw character data to integers.
    sint_data = np.frombuffer(origin_data, dtype=np.short)
    # Find the maximum volume and use it to normalize.
    data_max = max(abs(sint_data))
    print("data max: ", data_max)
    # Normalize and bigger 1.5
    int_data = sint_data * 1.0/data_max
    # Carrier frequency of 30khz.
    carrier = wave.open("output_carrier.wav", "w")
    # AM-SC about f=30khz.
    amsc = wave.open("output_amsc.wav", "w")
    # The final result of am modulation.
    am = wave.open("output_am.wav", "w")
    # Set audio parameters.
    for f in [am,amsc,carrier]:
        f.setnchannels(1) # Set to single channels.
        f.setsampwidth(2) # Set to 16-bit audio.
        f.setframerate(96000) # Set to 96khz sample rate.

    # Traverse all audio data.
    for n in range(0, data_len):
        # Must: Carrier frequency - voice frequency > 20k
        # Generate a single sample of the carrier signal.
        carrier_sample = math.cos(25000 * (n / 96000) * math.pi * 2)
        # Generate a single sample of the AM-SC.
        signal_amsc =  int_data[n] * carrier_sample
        # Generate a single sample of the final result.
        signal_am = (int_data[n] * carrier_sample + carrier_sample) / 2

        # Store to file.
        am.writeframes(struct.pack('h', int(signal_am * data_max)))
        amsc.writeframes(struct.pack('h', int(signal_amsc * data_max)))
        carrier.writeframes(struct.pack('h', int(carrier_sample * data_max)))


if __name__=='__main__':
    main()