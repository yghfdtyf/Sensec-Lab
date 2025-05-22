# test_single.py
import numpy as np
import librosa
import torch
import torch.nn as nn
from pydub import AudioSegment
import os
import argparse
import warnings



# 模型定义（必须与训练时完全一致）
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x).view(-1, 128)
        return self.fc(x)

# 特征提取函数（保持与训练一致）
def extract_spectral_entropy(audio, sr, patch_size=5, stride=2):
    n_fft = 512
    hop_length = 137  # 确保与训练参数一致
    
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='blackman')
    S = np.abs(S)
    
    # 计算熵矩阵
    rows, cols = S.shape
    rows_ep = (rows - patch_size) // stride + 1
    cols_ep = (cols - patch_size) // stride + 1
    E_p = np.zeros((rows_ep, cols_ep))
    
    for i in range(rows_ep):
        for j in range(cols_ep):
            x = i * stride
            y = j * stride
            patch = S[x:x+patch_size, y:y+patch_size]
            power = np.square(patch)
            total = np.sum(power) + 1e-10
            entropy_val = -np.sum((power/total) * np.log(power/total + 1e-10))
            E_p[i, j] = entropy_val
    return E_p

# 音频预处理管道
def audio_preprocess(file_path):
    try:
        # 支持多种音频格式
        audio = AudioSegment.from_file(file_path)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != SAMPLING_RATE:
            audio = audio.set_frame_rate(SAMPLING_RATE)
            
        # 转换为numpy数组
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples /= np.iinfo(audio.array_type).max  # 归一化到[-1, 1]
        
        return samples, SAMPLING_RATE
    except Exception as e:
        print(f"文件加载失败: {str(e)}")
        return None, None

# 特征提取主函数
def extract_features(file_path):
    # 加载并预处理音频
    audio, sr = audio_preprocess(file_path)
    if audio is None:
        return []
    
    # 分割音频段
    segment_samples = int(SEGMENT_LENGTH * sr)
    features = []
    
    for ptr in range(0, len(audio), segment_samples):
        segment = audio[ptr:ptr+segment_samples]
        if len(segment) < segment_samples:
            continue  # 丢弃不足300ms的末尾
        
        # 计算频谱混沌特征
        try:
            feature = extract_spectral_entropy(segment, sr)
            features.append(feature)
        except Exception as e:
            print(f"特征提取失败: {str(e)}")
            continue
    
    return features

# 模型加载与预测
class AudioClassifier:
    def __init__(self, model_path):
        self.model = SimpleCNN().to(DEVICE)
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            self.model = None
    
    def predict(self, features, threshold=0.5):
        if self.model is None:
            return None
        
        # 转换为模型输入格式
        X = np.array(features)[:, np.newaxis]  # 添加通道维度
        tensor_X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        
        # 批量预测AudioClassifier
        with torch.no_grad():
            outputs = self.model(tensor_X)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 攻击类概率
        
        return {
            'segment_probs': probs.cpu().numpy(),
            'average_prob': float(probs.mean()),
            'is_attack': bool(probs.mean() > threshold)
        }




# 配置与训练完全一致的参数
SEGMENT_LENGTH = 0.3  # 300ms
SAMPLING_RATE = 48000  # 必须48kHz
MODEL_PATH = "./path/audio_classifier_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 禁用不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='单音频文件分类检测')
# parser.add_argument('file_path', default="./speech_record.mp3",help='输入音频文件路径')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='分类阈值 (默认0.5)')
args = parser.parse_args()
args.file_path = "./temp/speech_record.wav"
# 初始化分类器
classifier = AudioClassifier(MODEL_PATH)
if classifier.model is None:
    exit(1)



# if __name__ == "__main__":
def start_ultrasonic_detect():


    # 特征提取
    features = extract_features(args.file_path)
    if not features:
        print("错误：未提取到有效特征，请检查：")
        print(f"1. 文件长度是否≥{SEGMENT_LENGTH}s")
        print(f"2. 采样率是否为{SAMPLING_RATE/1000}kHz")
        print("3. 文件格式是否支持（WAV/MP3/FLAC等）")
        exit(1)

    # 执行预测
    result = classifier.predict(features, args.threshold)
    
    # 打印结果
    print("\n检测结果:")
    print(f"文件路径: {args.file_path}")
    print(f"音频分段数: {len(features)}")
    print(f"平均攻击概率: {result['average_prob']:.4f}")
    print(f"最终判定: {'攻击音频' if result['is_attack'] else '正常音频'}")
    
    # 详细概率输出
    print("\n各分段详细概率:")
    for i, prob in enumerate(result['segment_probs']):
        print(f"分段 {i+1}: {prob:.4f} | 判定: {'攻击' if prob > args.threshold else '正常'}")

    if result['is_attack']:
        return True
    else:
        return False

if __name__=="__main__":
    start_ultrasonic_detect()