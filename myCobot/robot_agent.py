# 导入常用函数
from utils.asr import *             # 录音+语音识别
from utils.robot import *           # 连接机械臂
from utils.llm import *             # 大语言模型API
from utils.led import *             # 控制LED灯颜色
from utils.camera import *          # 摄像头
from utils.robot import *           # 机械臂运动
from utils.pump import *            # GPIO、夹爪
from utils.vlm_move import *        # 多模态大模型识别图像，夹爪抓取并移动物体
from utils.drag_teaching import *   # 拖动示教
from utils.agent import *           # 智能体Agent编排
from utils.tts import TextToSpeech, AudioPlayer

# 使用 AudioPlayer 播放音频
AudioPlayer.play_wav('asset/welcome.wav')


def agent_play():
    '''
    主函数，语音控制机械臂智能体编排动作
    '''
    # 归零
    back_zero()
    
    # print('测试摄像头')
    # check_camera()
    
    # # 输入指令
    # start_record_ok = print('现在进入等待激活模式(number;k)\n')
    # time.sleep(1)
    # if str.isnumeric(start_record_ok):
    #     DURATION = int(start_record_ok)
    #     record(DURATION=DURATION)   # 录音
    #     order = speech_recognition() # 语音识别
    # elif start_record_ok == 'k':
    #     order = input('请输入指令') # test
    #     print("您的指令是： " + order,'\n')
    #     print("马上开始执行!!!")
    # else:
    #     print('无指令，退出')
    #     # exit()
    #     raise NameError('无指令，退出')

    # 输入指令
    # 先回到原点，再把LED灯改为墨绿色，然后把绿色方块放在篮球上
    start_record_ok = input('是否开启录音，输入数字录音指定时长，按k打字输入，按c输入默认指令\n')
    if str.isnumeric(start_record_ok):
        DURATION = int(start_record_ok)
        record(DURATION=DURATION)   # 录音
        order = speech_recognition() # 语音识别
    elif start_record_ok == 'k':
        order = input('请输入指令')
    elif start_record_ok == 'c':
        order = '把蓝色方块放在红色方块上'
        # order = '先归零，再摇头，然后把蓝色方块放在红色方块上'
    else:
        print('无指令，退出')
        # exit()
        raise NameError('无指令，退出')
    
    # 智能体Agent编排动作
    agent_plan_output = agent_plan(order)
    print('智能编排动作如下:', agent_plan_output)
    for each in agent_plan_output: # 运行智能体规划编排的每个函数
        print('开始执行动作', each)
        eval(each)

# agent_play()
if __name__ == '__main__':
    agent_play()
