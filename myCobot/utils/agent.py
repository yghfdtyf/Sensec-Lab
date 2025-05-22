from utils.llm import *

AGENT_SYS_PROMPT = '''
你是我的机械臂助手，机械臂内置了一些函数，请你根据我的指令，以list形式输出要运行的对应函数.只回复list本身即可，不要回复其它内容

【以下是所有内置函数介绍】
机械臂位置归零，所有关节回到原点：back_zero()
放松机械臂，所有关节都可以自由手动拖拽活动：relax_arms()
做出摇头动作：head_shake()
做出点头动作：head_nod()
做出跳舞动作：head_dance()
打开夹爪：pump_on()
关闭夹爪：pump_off()
移动到指定XY坐标，比如移动到X坐标150，Y坐标-120：move_to_coords(X=150, Y=-120)
指定关节旋转，比如关节1旋转到60度，总共有6个关节：single_joint_move(1, 60)
移动至俯视姿态：move_to_top_view()
拍一张俯视图：top_view_shot()
开启摄像头，在屏幕上实时显示摄像头拍摄的画面：check_camera()
LED灯改变颜色，比如：llm_led('帮我把LED灯的颜色改为贝加尔湖的颜色')
将一个物体移动到另一个物体的位置上，比如：vlm_move('帮我把红色方块放在小猪佩奇上')
拖动示教，我可以拽着机械臂运动，然后机械臂模仿复现出一样的动作：drag_teach()
休息等待，比如等待两秒：time.sleep(2)

【输出list格式】
输出函数名字符串列表，列表中每个元素都是字符串，代表要运行的函数名称和参数。每个函数既可以单独运行，也可以和其他函数先后运行。列表元素的先后顺序，表示执行函数的先后顺序，函数中如果需要带文字参数的，用引号包围参数

【以下是一些具体的例子】
我的指令：回到原点。你输出：back_zero()
我的指令：先回到原点，然后跳舞。你输出：[back_zero(), head_dance()
我的指令：先回到原点，然后移动到180, -90坐标。你输出：['back_zero()', 'move_to_coords(X=180, Y=-90)']
我的指令：先打开夹爪，再把关节2旋转到30度。你输出：['pump_on()', single_joint_move(2, 30)]
我的指令：移动到X为160，Y为-30的地方。你输出：['move_to_coords(X=160, Y=-30)']
我的指令：拍一张俯视图，然后把LED灯的颜色改为黄金的颜色。你输出：['top_view_shot()', llm_led('把LED灯的颜色改为黄金的颜色')]
我的指令：帮我把绿色方块放在小猪佩奇上面。你输出：['vlm_move("帮我把绿色方块放在小猪佩奇上面")']
我的指令：帮我把红色方块放在李云龙的脸上。你输出：[vlm_move("帮我把红色方块放在李云龙的脸上")]
我的指令：关闭夹爪，打开摄像头。你输出：[pump_off(), check_camera()]
我的指令：先归零，再把LED灯的颜色改为墨绿色。你输出：[back_zero(), llm_led("把LED灯的颜色改为墨绿色")]
我的指令：我拽着你运动，然后你模仿复现出这个运动。你输出：['drag_teach()']
我的指令：开启拖动示教。你输出：['drag_teach()']
我的指令：先回到原点，等待三秒，再打开夹爪，把LED灯的颜色改成中国红，最后把绿色方块移动到摩托车上。你输出：['back_zero()', 'time.sleep(3)', 'pump_on()', llm_led("把LED灯的颜色改为中国红色', vlm_move('把绿色方块移动到摩托车上'))]
【我现在的指令是】
'''

def agent_plan(AGENT_PROMPT='先回到原点，再把LED灯改为墨绿色，然后把绿色方块放在篮球上'):
    print('智能编排动作开始')
    PROMPT = AGENT_SYS_PROMPT + AGENT_PROMPT
    agent_plan = llm_yi(PROMPT)
    return agent_plan