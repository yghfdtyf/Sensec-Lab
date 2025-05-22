import os
import sys
import json
import pyaudio
import playsound
from robot_agent import *
from vosk import Model, KaldiRecognizer



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




if __name__ == '__main__':
    # 1. 加载 Vosk 英文模型
    model_path = "path/vosk-model-small-en-us-0.15"  # 替换为英文模型路径
    if not os.path.exists(model_path):
        print("请下载并解压 Vosk 英文模型: https://alphacephei.com/vosk/models")
        sys.exit(1)
    model = Model(model_path)

    # 2. 初始化音频流参数
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = supported_rates
    CHUNK = 4000
    SILENCE_THRESHOLD = 0.01  # 静音阈值，可根据环境调整
    KEYWORDS = {"hi", "hello","hey"}  # 英文唤醒词列表

    # 4. 优化后的识别循环
    while True:
        # 初始化音频流
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index= card1_index  # 使用默认设备
        )

        # 3. 初始化识别器
        recognizer = KaldiRecognizer(model, RATE)
        recognizer.SetWords(True)  # 获取单词时间戳（可选）

        print("语音唤醒已启动，等待唤醒词...")

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if len(data) == 0:
                continue

            # 进行语音识别
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()  # 转换为小写
                print(f"完整识别结果: {text}")

                # 检查唤醒词
                if any(keyword in text.split() for keyword in KEYWORDS):
                    # 清理资源
                    stream.stop_stream()
                    stream.close()
                    audio.terminate()
                    time.sleep(0.1)  # 给操作系统释放时间



                    agent_play_secure()

                    break

            else:
                # 获取部分结果用于实时反馈
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "").lower()
                if partial_text:
                    print(f"当前输入: {partial_text}")
