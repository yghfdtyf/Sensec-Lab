#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/30/周一 16:23
# @Author  : FangFano
# @File    : mp32wav.py
# @Software: PyCharm

from pydub import AudioSegment

def mp3_to_wav(wav_path, result_path):
    # load .mp3 file
    audio = AudioSegment.from_wav(wav_path)

    # convert to single channel
    single_audio = audio.set_channels(1)

    # output as .wav file
    single_audio.export(result_path, format="wav")

# original .mp3 file
wav_path = "origin.wav"
# target .wav file
result_path = "1_single.wav"

mp3_to_wav(wav_path, result_path)

