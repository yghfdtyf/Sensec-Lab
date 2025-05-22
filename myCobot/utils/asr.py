
import pyaudio
import wave
import numpy as np
import os
import sys


# Initialize audio once
audio = pyaudio.PyAudio()
# Find card 1's index (run this once)
card1_index = 0
for i in range(audio.get_device_count()):
    dev_info = audio.get_device_info_by_index(i)
    if dev_info['maxInputChannels'] > 0 and "USB ENC Audio Device" in dev_info['name']:
        card1_index = i
        print("USB ENC Audio Device is: ", card1_index)
        break
if card1_index == 0:
    print("USB ENC Audio Device not found!")
# 获取设备信息

dev_info = audio.get_device_info_by_index(card1_index)
supported_rates = int(dev_info.get("defaultSampleRate", 16000))






def record(DURATION=5):
    '''
    调用麦克风录音，需用arecord -l命令获取麦克风ID
    DURATION，录音时长
    '''
    # print('开始 {} 秒录音'.format(DURATION))
    print('开始录音')
    os.system('arecord -D "plughw:{}" -f dat -c 1 -r 16000 -d {} temp/speech_record.wav'.format(card1_index, DURATION))
    print('录音结束')



def record_auto(MIC_INDEX=1):
    '''
    开启麦克风录音，保存至'temp/speech_record.wav'
    音量超过阈值自动开始录音，低于阈值一段时间后自动停止
    MIC_INDEX：麦克风设备索引号
    '''
    os.environ['PYAUDIO_IGNORE_ALSA_WARNINGS'] = '1'  # 抑制ALSA警告

    # 获取设备参数
    p = pyaudio.PyAudio()
    
    try:
        # 验证设备有效性
        device_info = p.get_device_info_by_index(MIC_INDEX)
        if device_info['maxInputChannels'] < 1:
            raise ValueError("该设备不是有效的输入设备")

        # 动态获取参数（兼容不同设备）
        RATE = int(device_info.get('defaultSampleRate', 16000))
        CHANNELS = int(device_info.get('maxInputChannels', 1))
        print(f"使用音频设备: {device_info['name']} | 采样率: {RATE}Hz | 通道数: {CHANNELS}")

    except Exception as e:
        print(f"设备参数错误: {str(e)}")
        print("\n可用输入设备列表：")
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                print(f"[{i}] {dev['name']} (Channels: {dev['maxInputChannels']})")
        p.terminate()
        return

    # 录音参数（保持不变）
    FORMAT = pyaudio.paInt16
    CHUNK = 1024
    QUIET_THRESHOLD = int(0.05 * 32767)
    SILENCE_TIMEOUT = 2
    MAX_DURATION = 30

    try:
        # 初始化录音流（增加异常捕获）
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=MIC_INDEX,
            start=False  # 避免立即启动流
        )
    except Exception as e:
        print(f"无法打开音频流: {str(e)}")
        p.terminate()
        return
    

    # 状态控制逻辑（保持不变）
    # 状态变量
    frames = []
    is_recording = False
    last_loud_time = 0
    start_frame = 0
    current_frame = 0
    silence_frames = int(SILENCE_TIMEOUT * RATE / CHUNK)
    max_frames = int(MAX_DURATION * RATE / CHUNK)

    print('等待语音输入...')

    while True:
        # 读取音频数据
        data = stream.read(CHUNK, exception_on_overflow=False)
        current_frame += 1
        
        # 计算当前块音量
        audio_data = np.frombuffer(data, dtype=np.int16)
        current_volume = np.max(np.abs(audio_data))

        # 状态机控制
        if not is_recording:
            if current_volume > QUIET_THRESHOLD:
                print("检测到语音，开始录音")
                is_recording = True
                start_frame = max(0, current_frame - 2)  # 包含触发前2个块
                last_loud_time = current_frame
                frames = [data]  # 重置存储
        else:
            frames.append(data)
            
            
            # 更新最后有效时间
            if current_volume > QUIET_THRESHOLD:
                last_loud_time = current_frame

            # 停止条件检测
            if (current_frame - last_loud_time) > silence_frames:
                print(f"静音超时{SILENCE_TIMEOUT}秒，停止录音")
                break 
 
            if current_frame > max_frames:
                print("达到最大录音时长，停止录音")
                break


    if 'stream' in locals() and stream.is_active():
        stream.stop_stream()
        stream.close()
    p.terminate()

    # 文件保存逻辑（保持不变）
    # 保存有效录音段
    end_frame = min(last_loud_time + silence_frames, current_frame)
    valid_frames = frames[:end_frame - start_frame]

    # 写入文件
    output_path = 'temp/speech_record.wav'
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(valid_frames))
    
    print(f'录音已保存至 {output_path}')
    return output_path


from API_KEY import *
import appbuilder
import soundfile as sf
import librosa


# 配置密钥
os.environ["APPBUILDER_TOKEN"] = APPBUILDER_TOKEN
asr = appbuilder.ASR() # 语音识别组件
def speech_recognition(audio_path='./temp/speech_record.wav'):
    '''
    AppBuilder-SDK语音识别组件
    '''
    print('开始语音识别')

    # audio_path_16k = './temp/speech_record_16k.wav'
    # # 直接读取原始数据及采样率
    # data, sr = sf.read(audio_path)  # 自动识别48kHz 
    # # 重采样
    # data_16k = librosa.resample(data.T, orig_sr=sr, target_sr=16000)  # 转置处理多声道 
    # # 保存（自动处理多声道）
    # sf.write(audio_path_16k, data_16k.T, 16000)  # 


    # 载入wav音频文件
    with wave.open(audio_path, 'rb') as wav_file:
        
        # 获取音频文件的基本信息
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        
        # 获取音频数据
        frames = wav_file.readframes(num_frames)
        
    # 向API发起请求
    content_data = {"audio_format": "wav", "raw_audio": frames, "rate": 16000}
    message = appbuilder.Message(content_data)
    # print(message)
    speech_result = asr.run(message).content['result'][0]
    print('语音识别结果：', speech_result)
    return speech_result
