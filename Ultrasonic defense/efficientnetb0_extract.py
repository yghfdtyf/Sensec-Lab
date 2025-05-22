import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import efficientnet_b0  
import torch.nn.functional as F
from pydub import AudioSegment
import random
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
# 参数设置
SEGMENT_LENGTH = 0.3  # 300ms
SAMPLING_RATE = 48000  # 48kHz采样率
DEBUG_SPLIT = 0
CACHE_FILE = "./data_chaotic_map.npy"
SAVE_PATH = "micguard_dsvdd.pth"
FEATURE_SAVE_DIR ="./efficientnetb0_feature"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特征提取函数（二维频谱混沌图，参考论文4.2节）
def extract_spectral_entropy(audio, sr, patch_size=5, stride=3):
    """计算频谱混沌图"""
    n_fft = 512
    hop_length = 375
    window = 'blackman'
    
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
    S = np.abs(S)
    
    rows, cols = S.shape
    rows_ep = (rows - patch_size) // stride + 1
    cols_ep = (cols - patch_size) // stride + 1
    E_p = np.zeros((rows_ep, cols_ep))
    
    for i, x in enumerate(range(0, rows - patch_size + 1, stride)):
        for j, y in enumerate(range(0, cols - patch_size + 1, stride)):
            patch = S[x:x+patch_size, y:y+patch_size]
            power = np.square(patch)
            power = power / (np.sum(power) + 1e-10)
            entropy_val = -np.sum(power * np.log(power + 1e-10))
            E_p[i, j] = entropy_val
    
    return E_p

def extract_features(audio_input):
    """提取二维频谱混沌图特征"""
    if isinstance(audio_input, AudioSegment):
        audio = audio_input
    else:
        audio = AudioSegment.from_file(audio_input)
    
    y = np.array(audio.get_array_of_samples()).astype(np.float32) / np.iinfo(audio.array_type).max
    sr = audio.frame_rate
    features = []
    
    for i in range(0, len(y), int(SEGMENT_LENGTH * sr)):
        segment = y[i:i + int(SEGMENT_LENGTH * sr)]
        if len(segment) < int(SEGMENT_LENGTH * sr):
            continue
        chaotic = extract_spectral_entropy(segment, sr)
        features.append(chaotic)
    
    return features

# 数据准备
def prepare_dataset(normal_dir, attack_dir):
    X, y = [], []
    
    # 预加载正常音频
    normal_segments_cache = {}
    print("预加载正常音频...")
    for file in tqdm(os.listdir(normal_dir)):
        if file.endswith('.wav'):
            audio = AudioSegment.from_wav(os.path.join(normal_dir, file))
            if DEBUG_SPLIT:
                audio = audio[:DEBUG_SPLIT * 1000]
            step = int(SEGMENT_LENGTH * 1000)
            segments = [audio[start:start+step] for start in range(0, len(audio), step)
                        if len(audio[start:start+step]) >= step]
            normal_segments_cache[file] = segments
            features = extract_features(audio)
            X.extend(features)
            y.extend([0] * len(features))
    
    # 预加载攻击音频
    attack_audio_cache = {}
    print("预加载攻击音频...")
    for file in tqdm(os.listdir(attack_dir)):
        if file.endswith('.wav'):
            attack_audio_cache[file] = AudioSegment.from_wav(os.path.join(attack_dir, file))
            if DEBUG_SPLIT:
                attack_audio_cache[file] = attack_audio_cache[file][:DEBUG_SPLIT * 1000]
    
    # 处理混合音频
    def process_mixed_audio(attack_duration_range, label):
        attack_files = list(attack_audio_cache.keys())
        if not attack_files:
            return
        for normal_file, segments in tqdm(normal_segments_cache.items()):
            for normal_segment in segments:
                attack_file = random.choice(attack_files)
                attack_audio = attack_audio_cache[attack_file]
                mixed_segments = []
                attack_duration = random.randint(*attack_duration_range)
                
                if len(attack_audio) >= attack_duration:
                    attack_start = random.randint(0, len(attack_audio) - attack_duration)
                    attack_segment = attack_audio[attack_start:attack_start+attack_duration]
                    positions = random.sample(
                        range(0, len(normal_segment) - attack_duration),
                        min(4, (len(normal_segment) - attack_duration) // 10)
                    )
                    for pos in positions:
                        mixed = normal_segment.overlay(attack_segment, position=pos)
                        mixed_segments.append(mixed)
                
                if mixed_segments:
                    combined = sum(mixed_segments)
                    features = extract_features(combined)
                    X.extend(features)
                    y.extend([label] * len(features))
    
    # print("处理短时混合音频(<=20ms)...")
    # process_mixed_audio((1, 20), 0)
    # print("处理长时混合音频(21-290ms)...")
    # process_mixed_audio((21, 290), 1)
    
    print("处理纯攻击音频...")
    for attack_audio in tqdm(attack_audio_cache.values()):
        features = extract_features(attack_audio)
        X.extend(features)
        y.extend([1] * len(features))
    
    X = np.array(X)  # 形状为 (样本数, rows_ep, cols_ep)
    y = np.array(y)
    data_dict = {'X': X, 'y': y}
    np.save(CACHE_FILE, data_dict)
    return X, y

class EfficientNetB0FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载 EfficientNetB0，不使用预训练权重，因为输入通道将被修改
        self.model = EfficientNet.from_name('efficientnet-b0')
        
        # 修改第一个卷积层以接受单通道输入（从 3 通道改为 1 通道）
        self.model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # 用于存储 MBConv 块的特征图
        self.features = []
        
        # 注册前向钩子以捕获 MBConv 块的输出
        self._register_hooks()

    def _register_hooks(self):
        # 定义钩子函数，将 MBConv 块的输出添加到 features 列表
        def hook(module, input, output):
            self.features.append(output)
        
        # 在所有 MBConv 块上注册钩子
        for name, module in self.model.named_modules():
            if 'blocks' in name and 'MBConv' in module.__class__.__name__:
                module.register_forward_hook(hook)

    def forward(self, x):
        # 清空之前的特征图
        self.features = []
        # 前向传播，触发钩子捕获特征
        _ = self.model(x)
        
        # 检查是否捕获到特征图
        if not self.features:
            raise ValueError("未捕获到任何特征图，请检查钩子注册是否正确。")
        
        # 获取第一个特征图的空间尺寸作为目标尺寸
        target_size = self.features[0].shape[2:]
        
        # 将所有特征图调整到统一尺寸并存储
        resized_features = []
        for feature in self.features:
            resized = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(resized)
        
        # 沿着通道维度拼接所有特征图
        concatenated_features = torch.cat(resized_features, dim=1)
        return concatenated_features

def save_features(features, labels, save_dir):
    """保存特征与标签到.npz文件（支持逐个样本处理后的合并）"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "features.npz")
    np.savez(
        save_path,
        features=features,
        labels=labels
    )

if __name__ == "__main__":
    # 加载预处理数据（保持原样）
    if os.path.exists(CACHE_FILE):
        data_dict = np.load(CACHE_FILE, allow_pickle=True).item()
        X, y = data_dict['X'], data_dict['y']
    else:
        X, y = prepare_dataset('normal', 'attack')

    # 数据预处理（保持原样）
    X = np.expand_dims(X, axis=1).astype(np.float32)
    # X = np.repeat(X, 3, axis=1)

    # 转换为PyTorch张量
    X_tensor = torch.tensor(X).float().to(DEVICE)
    y_tensor = torch.tensor(y).long().to(DEVICE)

    # 创建DataLoader
    batch_size = 32
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 初始化特征提取器
    feature_extractor = EfficientNetB0FeatureExtractor().to(DEVICE)
    feature_extractor.eval()

    # 修改后的特征提取逻辑（逐个样本处理）
    all_features = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Processing batches"):
            # 批量计算提高效率
            print(inputs.shape)
            batch_features = feature_extractor(inputs)  # [B, C, H, W]
            print(batch_features.shape)
            # 转换为CPU数据
            batch_features_cpu = batch_features.cpu().numpy()
            batch_labels_cpu = labels.cpu().numpy()
            
            # 逐个样本处理
            for i in range(batch_features.size(0)):
                single_feature = batch_features_cpu[i]  # [C, H, W]
                all_features.append(single_feature)
                all_labels.append(batch_labels_cpu[i])

    # 合并所有样本
    all_features = np.stack(all_features, axis=0)  # [N, C, H, W]
    all_labels = np.array(all_labels)

    # 保存特征
    save_features(all_features, all_labels, FEATURE_SAVE_DIR)
    print(f"特征已保存至 {FEATURE_SAVE_DIR}/features.npz")
    print(f"最终特征形状：{all_features.shape}，标签形状：{all_labels.shape}")