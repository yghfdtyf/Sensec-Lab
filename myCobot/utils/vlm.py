# utils_vlm.py
# 2024-5-22
# 多模态大模型、可视化

# print('导入VLM大模型模块')
import time
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont, ImageDraw

import json
# 导入中文字体，指定字号
font = ImageFont.truetype('asset/SimHei.ttf', 26)

from API_KEY import *

# 系统提示词
SYSTEM_PROMPT = '''
我即将说一句给机械臂的指令，你帮我从这句话中提取出起始物体和终止物体，并从这张图中分别找到这两个物体左上角和右下角的像素坐标，输出json数据结构。

例如，如果我的指令是：请帮我把物体A放在物体B上。
请按以下严格格式生成JSON，注意坐标点要求：
{
  "start": "物体名称",
  "start_xyxy": [[x1,y1], [x2,y2]],  // 两个坐标点，每个点用二维数组表示
  "end": "物体名称",
  "end_xyxy": [[x1,y1], [x2,y2]]   // 例如：[[102,505], [324,860]]
}

注意：
1. 坐标必须是二维数组，不允许合并成四维数组。
2. 每个物体的xyxy字段必须包含两个坐标点（左上角和右下角）。
3. 只回复json本身即可，不要回复其它内容.

我现在的指令是：
'''

# Yi-Vision调用函数
from openai import OpenAI
import base64
import httpx

def yi_vision_api(PROMPT='帮我把红色方块放在钢笔上', img_path='temp/vl_now.jpg'):
    
    client = OpenAI(
        api_key= QWEN_KEY,
        base_url= QWEN_URL,
    )
    
    # 编码为base64数据
    with open(img_path, 'rb') as image_file:
        image = 'data:image/jpeg;base64,' + base64.b64encode(image_file.read()).decode('utf-8')
    
    # 向大模型发起请求
    completion = client.chat.completions.create(
      model= QWEN_MODEL,
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": SYSTEM_PROMPT + PROMPT
            },
            {
              "type": "image_url",
              "image_url": {
                "url": image
              }
            }
          ]
        },
      ]
    )
    
    # 解析大模型返回结果
    response_content = completion.choices[0].message.content.strip()
    # print(f"[DEBUG] 原始响应内容:\n{response_content}")  # 打印原始内容
    cleaned_content = response_content.replace("```json", "").replace("```", "").strip()
    # print(f"[DEBUG] 清洗后内容:\n{cleaned_content}")    # 打印清洗后内容
    result = json.loads(cleaned_content)
    print('大模型调用成功！')
    return result

# 添加 OpenRouter API 调用函数
def openrouter_vision_api(PROMPT='帮我把红色方块放在钢笔上', img_path='temp/vl_now.jpg', model=None):
    """使用 OpenRouter API 处理多模态图片分析请求"""
    
    # 从 API_KEY.py 导入配置
    from API_KEY import OPENROUTER_API_KEY, OPENROUTER_URL, OPENROUTER_MODEL
    
    # 使用指定模型或默认模型
    API_MODEL = model if model else OPENROUTER_MODEL
    
    # 创建 OpenAI 客户端，指向 OpenRouter
    client = OpenAI(
        base_url=OPENROUTER_URL,
        api_key=OPENROUTER_API_KEY,
        http_client=httpx.Client(proxy="socks5://192.168.174.1:10808")  # 明确禁用代理
    )
    
    # 编码图片为 base64 数据
    with open(img_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 重试机制
    for i in range(10):
        try:
            # 向大模型发起请求
            completion = client.chat.completions.create(
                model=API_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT + PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            # 解析大模型返回结果
            response_content = completion.choices[0].message.content.strip()
            cleaned_content = response_content.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(cleaned_content)
            print('OpenRouter大模型调用成功！')
            return result
            
        except Exception as e:
            print(f"OpenRouter API调用失败: {e}，第{i+1}次尝试")
            time.sleep(1)  # 失败后稍微等待再重试
            
            if i == 9:  # 最后一次尝试失败
                print("多次尝试后调用仍然失败")
                # 返回一个表示错误的默认结构
                return {
                    "error": str(e),
                    "start": "错误",
                    "start_xyxy": [[0, 0], [100, 100]],
                    "end": "错误",
                    "end_xyxy": [[0, 0], [100, 100]]
                }

def vision_api(PROMPT='帮我把红色方块放在钢笔上', img_path='temp/vl_now.jpg', api_provider='qwen', model=None):
    """统一的视觉API接口，支持不同提供商
    
    参数:
        PROMPT: 指令内容
        img_path: 图片路径
        api_provider: 'qwen' 或 'openrouter'
        model: 模型名称（仅用于OpenRouter，可选）
    
    返回:
        解析后的JSON结果
    """
    if api_provider.lower() == 'qwen':
        return yi_vision_api(PROMPT, img_path)
    elif api_provider.lower() == 'openrouter':
        return openrouter_vision_api(PROMPT, img_path, model)
    else:
        raise ValueError(f"不支持的API提供商: {api_provider}")

def post_processing_viz(result, img_path, check=False):
    
    '''
    视觉大模型输出结果后处理和可视化
    check：是否需要人工看屏幕确认可视化成功，按键继续或退出
    '''
    
    # 后处理
    img_bgr = cv2.imread(img_path)
    img_h = img_bgr.shape[0]
    img_w = img_bgr.shape[1]
    # 缩放因子
    FACTOR = 999
    # 起点物体名称
    START_NAME = result['start']
    # 终点物体名称
    END_NAME = result['end']
    # 起点，左上角像素坐标
    START_X_MIN = int(result['start_xyxy'][0][0])
    START_Y_MIN = int(result['start_xyxy'][0][1])
    # START_X_MIN = int(result['start_xyxy'][0][0] * img_w / FACTOR)
    # START_Y_MIN = int(result['start_xyxy'][0][1] * img_h / FACTOR)
    # 起点，右下角像素坐标
    START_X_MAX = int(result['start_xyxy'][1][0])
    START_Y_MAX = int(result['start_xyxy'][1][1])
    # START_X_MAX = int(result['start_xyxy'][1][0] * img_w / FACTOR)
    # START_Y_MAX = int(result['start_xyxy'][1][1] * img_h / FACTOR)
    # 起点，中心点像素坐标
    START_X_CENTER = int((START_X_MIN + START_X_MAX) / 2)
    START_Y_CENTER = int((START_Y_MIN + START_Y_MAX) / 2)
    # 终点，左上角像素坐标
    END_X_MIN = int(result['end_xyxy'][0][0])
    END_Y_MIN = int(result['end_xyxy'][0][1])
    # END_X_MIN = int(result['end_xyxy'][0][0] * img_w / FACTOR)
    # END_Y_MIN = int(result['end_xyxy'][0][1] * img_h / FACTOR)
    # 终点，右下角像素坐标
    END_X_MAX = int(result['end_xyxy'][1][0])
    END_Y_MAX = int(result['end_xyxy'][1][1])
    # END_X_MAX = int(result['end_xyxy'][1][0] * img_w / FACTOR)
    # END_Y_MAX = int(result['end_xyxy'][1][1] * img_h / FACTOR)
    # 终点，中心点像素坐标
    END_X_CENTER = int((END_X_MIN + END_X_MAX) / 2)
    END_Y_CENTER = int((END_Y_MIN + END_Y_MAX) / 2)
    
    # 可视化
    # 画起点物体框
    img_bgr = cv2.rectangle(img_bgr, (START_X_MIN, START_Y_MIN), (START_X_MAX, START_Y_MAX), [0, 0, 255], thickness=3)
    # 画起点中心点
    img_bgr = cv2.circle(img_bgr, [START_X_CENTER, START_Y_CENTER], 6, [0, 0, 255], thickness=-1)
    # 画终点物体框
    img_bgr = cv2.rectangle(img_bgr, (END_X_MIN, END_Y_MIN), (END_X_MAX, END_Y_MAX), [255, 0, 0], thickness=3)
    # 画终点中心点
    img_bgr = cv2.circle(img_bgr, [END_X_CENTER, END_Y_CENTER], 6, [255, 0, 0], thickness=-1)
    # 写中文物体名称
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # BGR 转 RGB
    img_pil = Image.fromarray(img_rgb) # array 转 pil
    draw = ImageDraw.Draw(img_pil)
    # 写起点物体中文名称
    draw.text((START_X_MIN, START_Y_MIN-32), START_NAME, font=font, fill=(255, 0, 0, 1)) # 文字坐标，中文字符串，字体，rgba颜色
    # 写终点物体中文名称
    draw.text((END_X_MIN, END_Y_MIN-32), END_NAME, font=font, fill=(0, 0, 255, 1)) # 文字坐标，中文字符串，字体，rgba颜色
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR) # RGB转BGR
    # 保存可视化效果图
    cv2.imwrite('temp/vl_now_viz.jpg', img_bgr)

    formatted_time = time.strftime("%Y%m%d%H%M", time.localtime())
    cv2.imwrite('visualizations/{}.jpg'.format(formatted_time), img_bgr)

    # 在屏幕上展示可视化效果图
    # cv2.imshow('result', img_bgr) 

    # if check:
    #     print('    请确认可视化成功，按c键继续，按q键退出')
    #     while(True):
    #         key = cv2.waitKey(10) & 0xFF
    #         if key == ord('c'): # 按c键继续
    #             break
    #         if key == ord('q'): # 按q键退出
    #             # exit()
    #             cv2.destroyAllWindows()   # 关闭所有opencv窗口
    #             raise NameError('按q退出')
    # else:
    #     if cv2.waitKey(1) & 0xFF == None:
    #         pass

    return START_X_CENTER, START_Y_CENTER, END_X_CENTER, END_Y_CENTER

# 如果直接运行此文件，则执行简单测试
if __name__ == "__main__":
    import os
    
    # 确保测试图片存在
    test_img_path = 'temp/vl_now.jpg'
    if not os.path.exists(test_img_path):
        print(f"警告: 测试图片 {test_img_path} 不存在，将使用示例图片")
        # 你可以选择使用项目中的其他图片进行测试
        # 或者复制一个现有图片到所需位置
        
        # 替代方案: 查找项目中可能存在的测试图片
        test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')
        available_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        if available_images:
            test_img_path = os.path.join(test_dir, available_images[0])
            print(f"使用备选图片: {test_img_path}")
    
    test_prompt = "帮我把红色方块放在蓝色方块上"
    
    # 测试通义千问 Vision API
    print("\n===== 测试通义千问 Vision API =====")
    try:
        result_qwen = vision_api(test_prompt, img_path=test_img_path, api_provider='qwen')
        print(f"测试结果: {json.dumps(result_qwen, ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"通义千问测试失败: {e}")
    
    # 测试 OpenRouter Vision API
    print("\n===== 测试 OpenRouter Vision API =====")
    try:
        result_openrouter = vision_api(test_prompt, img_path=test_img_path, api_provider='openrouter')
        print(f"测试结果: {json.dumps(result_openrouter, ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"OpenRouter测试失败: {e}")
