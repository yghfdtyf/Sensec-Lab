import ast
import time
import json
from openai import OpenAI
from API_KEY import *

def llm_yi(PROMPT='你好，你是谁？'):
    """调用通义千问大模型API"""
    
    API_BASE = QWEN_URL
    API_KEY = QWEN_KEY
    API_MODEL = QWEN_MODEL
    
    for i in range(10):
        # 访问大模型API
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_BASE)

            completion = client.chat.completions.create(
                model = API_MODEL,
                messages=[{"role": "user", "content": PROMPT}],
            )
            response_content = completion.choices[0].message.content.strip()
            print(f"[DEBUG] 原始响应内容:\n{response_content}")  # 打印原始内容
            
            # 清理响应内容 - 去除代码块标记和其他多余内容
            cleaned_content = response_content.replace("```json", "").replace("```python", "").replace("```", "").strip()
            print(f"[DEBUG] 清洗后内容:\n{cleaned_content}")  # 打印清洗后内容
            
            # 直接处理列表格式的响应
            if cleaned_content.startswith('[') and cleaned_content.endswith(']'):
                # 提取列表内容 (去除首尾的方括号)
                content_without_brackets = cleaned_content.strip('[]')
                
                # 如果内容为空，返回空列表
                if not content_without_brackets.strip():
                    result = []
                else:
                    # 初步按逗号分割
                    raw_items = []
                    
                    # 特殊处理函数调用，考虑函数内部可能有逗号
                    # 使用简单的状态机来解析
                    current_item = ""
                    in_parentheses = 0  # 跟踪括号嵌套层数
                    in_quotes = False   # 跟踪是否在引号内
                    escape_next = False # 处理转义字符
                    
                    for char in content_without_brackets:
                        if escape_next:
                            current_item += char
                            escape_next = False
                            continue
                            
                        if char == '\\':
                            current_item += char
                            escape_next = True
                            continue
                            
                        if char == '"' or char == "'":
                            in_quotes = not in_quotes
                            current_item += char
                        elif char == '(' and not in_quotes:
                            in_parentheses += 1
                            current_item += char
                        elif char == ')' and not in_quotes:
                            in_parentheses -= 1
                            current_item += char
                        elif char == ',' and not in_quotes and in_parentheses == 0:
                            # 只有在不在引号内且不在括号内时才分割
                            raw_items.append(current_item.strip())
                            current_item = ""
                        else:
                            current_item += char
                    
                    # 添加最后一项
                    if current_item.strip():
                        raw_items.append(current_item.strip())
                    
                    # 清理每个项目，去除多余的引号
                    result = []
                    for item in raw_items:
                        item = item.strip()
                        # 如果项目被单引号或双引号包围，去除它们
                        if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                            item = item[1:-1]
                        result.append(item)
            else:
                # 如果响应不是列表格式，则包装为单元素列表
                result = [cleaned_content]
            
            print("通义千问大模型调用成功!")
            break
                
        except Exception as e:
            print(f"通义千问API调用失败: {e}，第{i+1}次尝试")
            time.sleep(1)  # 失败后稍微等待再重试
            if i == 9:  # 最后一次尝试失败
                result = ["API调用失败"]  # 确保结果格式正确
            
    return result

def openrouter_llm(PROMPT='你好，你是谁？', model=None):
    """调用OpenRouter API"""
    
    API_BASE = OPENROUTER_URL
    API_KEY = OPENROUTER_API_KEY
    API_MODEL = OPENROUTER_MODEL if model is None else model
    
    for i in range(10):
        try:
            # 创建OpenAI客户端，指向OpenRouter
            client = OpenAI(
                base_url=API_BASE,
                api_key=API_KEY
            )
            
            completion = client.chat.completions.create(
                model=API_MODEL,
                messages=[{"role": "user", "content": PROMPT}],
            )
            
            # 检查响应是否有效
            if completion and hasattr(completion, 'choices') and completion.choices:
                response_content = completion.choices[0].message.content.strip()
                print(f"[DEBUG] OpenRouter原始响应:\n{response_content}")
                
                # 清理响应内容 - 去除代码块标记和其他多余内容
                cleaned_content = response_content.replace("```json", "").replace("```python", "").replace("```", "").strip()
                
                # 直接处理列表格式的响应，使用与llm_yi相同的解析逻辑
                if cleaned_content.startswith('[') and cleaned_content.endswith(']'):
                    # 提取列表内容
                    content_without_brackets = cleaned_content.strip('[]')
                    
                    # 如果内容为空，返回空列表
                    if not content_without_brackets.strip():
                        result = []
                    else:
                        # 初步按逗号分割
                        raw_items = []
                        
                        # 特殊处理函数调用
                        current_item = ""
                        in_parentheses = 0
                        in_quotes = False
                        escape_next = False
                        
                        for char in content_without_brackets:
                            if escape_next:
                                current_item += char
                                escape_next = False
                                continue
                                
                            if char == '\\':
                                current_item += char
                                escape_next = True
                                continue
                                
                            if char == '"' or char == "'":
                                in_quotes = not in_quotes
                                current_item += char
                            elif char == '(' and not in_quotes:
                                in_parentheses += 1
                                current_item += char
                            elif char == ')' and not in_quotes:
                                in_parentheses -= 1
                                current_item += char
                            elif char == ',' and not in_quotes and in_parentheses == 0:
                                raw_items.append(current_item.strip())
                                current_item = ""
                            else:
                                current_item += char
                        
                        # 添加最后一项
                        if current_item.strip():
                            raw_items.append(current_item.strip())
                        
                        # 清理每个项目
                        result = []
                        for item in raw_items:
                            item = item.strip()
                            if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                                item = item[1:-1]
                            result.append(item)
                else:
                    # 非列表格式
                    result = [cleaned_content]
                
                print("OpenRouter大模型调用成功!")
                break
            else:
                print(f"API返回无效响应")
                if i == 9:
                    result = ["API返回无效"]
                time.sleep(1)
                
        except Exception as e:
            print(f"OpenRouter API调用失败: {e}，第{i+1}次尝试")
            time.sleep(1)  # 失败后稍微等待再重试
            if i == 9:
                result = ["API调用失败"]
    
    # 确保在所有情况下都有返回值
    if 'result' not in locals():
        result = ["API调用未返回结果"]
    
    return result

# 统一LLM接口，支持选择不同提供商
def llm(PROMPT='你好，你是谁？', api_provider='qwen', model=None):
    """统一的LLM接口，支持不同提供商
    
    参数:
        PROMPT: 提示词
        api_provider: 'qwen' 或 'openrouter'
        model: 模型名称（仅用于OpenRouter，可选）
    
    返回:
        列表类型的模型输出结果
    """
    if api_provider.lower() == 'qwen':
        return llm_yi(PROMPT)
    elif api_provider.lower() == 'openrouter':
        return openrouter_llm(PROMPT, model)
    else:
        raise ValueError(f"不支持的API提供商: {api_provider}")

# 如果直接运行此文件，则执行简单测试
if __name__ == "__main__":
    # 测试通义千问
    print("\n===== 测试通义千问API =====")
    test_prompt = "请返回3个常用的问候语，格式为列表"
    result_qwen = llm(test_prompt, api_provider='qwen')
    print(f"测试结果: {result_qwen}")
    
    # 测试OpenRouter
    print("\n=====测试",OPENROUTER_MODEL,"======")
    result_openrouter = llm(test_prompt, api_provider='openrouter')
    print(f"测试结果: {result_openrouter}")
