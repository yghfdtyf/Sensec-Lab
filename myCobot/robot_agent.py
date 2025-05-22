# 导入常用函数
from utils.asr import *             # 录音+语音识别
from utils.robot import *           # 连接机械臂
from utils.llm import *             # 大语言模型API
from utils.led import *             # 控制LED灯颜色
from utils.camera import *          # 摄像头
from utils.pump import *            # GPIO、夹爪
from utils.vlm_move import *        # 多模态大模型识别图像，夹爪抓取并移动物体
# from utils.drag_teaching import *   # 拖动示教
from utils.agent import *           # 智能体Agent编排
from utils.tts import TextToSpeech, AudioPlayer
# from utils.ds_base_2D_test import *
from utils.ds_svm_test import *
import playsound

# 归零
back_zero()

# 播放welcome音频
# text_to_speech = TextToSpeech() # 创建 TextToSpeech 和 AudioPlayer 实例
# text_to_speech.synthesize_to_wav("你好,欢迎使用具身智能机械臂", output_path="temp/welcome.wav")
AudioPlayer.play_wav('temp/welcome.wav')


def agent_play():
    '''
    主函数，语音控制机械臂智能体编排动作
    '''

    # 输入指令
    # 先回到原点，再把LED灯改为墨绿色，然后把绿色方块放在篮球上
    start_record_ok = 'k'
    if start_record_ok == 'k':
        playsound.playsound("./temp/beep.mp3")  # 播放提示音  
        record(DURATION=10)   # 录音
        order = speech_recognition() # 语音识别
    elif start_record_ok == 'c':
        order = '把蓝色方块放在红色方块上'
    else:
        print('无指令，退出')
        return
    
    # text_to_speech.synthesize_to_wav("收到,开始智能编排", output_path='temp/Start_smart_orchestration.wav')
    # print('播放语音合成音频')
    AudioPlayer.play_wav('temp/Start_smart_orchestration.wav')

    # 智能体Agent编排动作
    agent_plan_output = agent_plan(order)
    print('智能编排动作如下:', agent_plan_output)
    # text_to_speech.synthesize_to_wav("动作编排完成,开始执行", output_path='temp/Finish_smart_orchestration.wav')
    AudioPlayer.play_wav('temp/Finish_smart_orchestration.wav')

    for each in agent_plan_output: # 运行智能体规划编排的每个函数
        print('开始执行动作', each)
        # 检查是否是vlm_move函数调用，并添加默认api_provider参数
        if each.startswith('vlm_move('):
            # 如果是简单的vlm_move调用没有指定api_provider
            if 'api_provider' not in each:
                # 在函数调用的最后一个括号前添加默认参数
                each = each.replace(')', ", api_provider='qwen')")
        eval(each)
    # text_to_speech.synthesize_to_wav("任务已完成", output_path='temp/task_finished.wav')
    AudioPlayer.play_wav('temp/task_finished.wav')


def agent_play_secure():
    '''
    主函数，语音控制机械臂智能体编排动作
    '''
    # 输入指令
    # 先回到原点，再把LED灯改为墨绿色，然后把绿色方块放在篮球上
    start_record_ok = 'k'
    if start_record_ok == 'k':
        playsound.playsound("./temp/beep.mp3")  # 播放提示音  
        record(DURATION=10)   # 录音

        # ultrasonic_detect
        # if(start_ultrasonic_detect()):
        if(not start_ultrasonic_detect_svm()):
            # text_to_speech.synthesize_to_wav("疑似受到超声波攻击,命令不会执行", output_path='temp/detect_ultrasonic.wav')
            # print('播放语音合成音频')
            AudioPlayer.play_wav('temp/detect_ultrasonic.wav')
            print('系统存在被超声波攻击的风险,命令不会执行')
            return
        order = speech_recognition() # 语音识别

    elif start_record_ok == 'c':
        order = '把蓝色方块放在红色方块上'
    else:
        print('无指令，退出')
        return
    
    # text_to_speech.synthesize_to_wav("收到,开始智能编排", output_path='temp/Start_smart_orchestration.wav')
    # print('播放语音合成音频')
    AudioPlayer.play_wav('temp/Start_smart_orchestration.wav')

    # 智能体Agent编排动作
    agent_plan_output = agent_plan(order)
    print('智能编排动作如下:', agent_plan_output)
    # text_to_speech.synthesize_to_wav("动作编排完成,开始执行", output_path='temp/Finish_smart_orchestration.wav')
    AudioPlayer.play_wav('temp/Finish_smart_orchestration.wav')
    for each in agent_plan_output: # 运行智能体规划编排的每个函数
        print('开始执行动作', each)
        eval(each)
    # text_to_speech.synthesize_to_wav("任务已完成", output_path='temp/task_finished.wav')
    AudioPlayer.play_wav('temp/task_finished.wav')
