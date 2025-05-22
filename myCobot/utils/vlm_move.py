from utils.robot import *
from utils.asr import *
from utils.vlm import *

import time

def vlm_move(PROMPT='帮我把绿色方块放在小猪佩奇上', input_way='keyboard', api_provider='qwen', model=None):
    '''
    多模态大模型识别图像，夹爪抓取并移动物体
    input_way：speech语音输入，keyboard键盘输入
    api_provider: 'qwen' 或 'openrouter'，选择API提供商
    model: 模型名称（仅用于OpenRouter，可选）
    '''

    print('多模态大模型识别图像，夹爪抓取并移动物体')
    
    # 机械臂归零
    print('机械臂归零')
    mc.send_angles([0, 0, 0, 0, 0, 0], 50)
    time.sleep(1)
    
    ## 第一步：完成手眼标定
    print('第一步：完成手眼标定')
    
    ## 第二步：发出指令
    print('第二步，给出的指令是：', PROMPT)
    
    ## 第三步：拍摄俯视图
    print('第三步：拍摄俯视图')
    top_view_shot(check=True)
    
    ## 第四步：将图片输入给多模态视觉大模型
    print('第四步：将图片输入给多模态视觉大模型')
    img_path = 'temp/vl_now.jpg'
    
    n = 1
    while n < 5:
        try:
            print(f'    尝试第 {n} 次访问多模态大模型（使用 {api_provider}）')
            result = vision_api(PROMPT, img_path=img_path, api_provider=api_provider, model=model)
            print('    多模态大模型调用成功！')
            break
        except Exception as e:
            print(f'    多模态大模型调用失败: {e}')
            n += 1
            if n >= 5:
                print('    多次尝试均失败，退出任务')
                return
    
    ## 第五步：视觉大模型输出结果后处理和可视化
    print('第五步：视觉大模型输出结果后处理和可视化')
    START_X_CENTER, START_Y_CENTER, END_X_CENTER, END_Y_CENTER = post_processing_viz(result, img_path, check=False)
    
    ## 第六步：手眼标定转换为机械臂坐标
    print('第六步：手眼标定，将像素坐标转换为机械臂坐标')
    # 起点，机械臂坐标
    START_X_MC, START_Y_MC = eye2hand(START_X_CENTER, START_Y_CENTER)
    # 终点，机械臂坐标
    END_X_MC, END_Y_MC = eye2hand(END_X_CENTER, END_Y_CENTER)
    
    ## 第七步：夹爪抓取移动物体
    print('第七步：夹爪抓取移动物体')
    pump_move(mc=mc, XY_START=[START_X_MC, START_Y_MC], XY_END=[END_X_MC, END_Y_MC])
    
    ## 第八步：收尾
    print('第八步：任务完成')
    cv2.destroyAllWindows()   # 关闭所有opencv窗口