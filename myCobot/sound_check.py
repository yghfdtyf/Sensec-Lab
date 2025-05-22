from utils.asr import *             # 录音+语音识别
from utils.tts import TextToSpeech, AudioPlayer

# 创建 TextToSpeech 和 AudioPlayer 实例
text_to_speech = TextToSpeech()

def main():
    print('开始录音5秒')
    record(DURATION=5)   # 录音
    # record_auto()
    print('播放录音')
    AudioPlayer.play_wav('temp/speech_record.wav')
    speech_result = speech_recognition()
    print('开始语音合成')
    text_to_speech.synthesize_to_wav(speech_result)
    print('播放语音合成音频')
    AudioPlayer.play_wav('temp/tts.wav')

if __name__ == "__main__":
    main()
