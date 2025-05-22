import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # 新增导入
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import wide_resnet50_2
import torch.nn.functional as F
from pydub import AudioSegment
import random
from tqdm import tqdm


# 参数设置
SEGMENT_LENGTH = 2.0 # 300ms
SAMPLING_RATE = 48000  # 48kHz采样率
DEBUG_SPLIT = 10
CACHE_FILE = "./data_chaotic_map.npy"
SAVE_PATH = "micguard_dsvdd.pth"
FEATURE_SAVE_DIR ="./feature"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 新增参数配置
GLOBAL_STATS_FILE = "./global_stats.npy"  # 全局统计量文件
VISUAL_SAVE_DIR = "./visualization"       # 可视化保存路径
# 特征提取函数（二维频谱混沌图，参考论文4.2节）
def extract_spectral_entropy(audio, sr, patch_size=4, stride=4):
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
        # print(chaotic.shape)
        # raise BaseException
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
            # print(audio.shape)
            
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
    # 修改后的混合音频处理函数
    def process_mixed_audio(attack_duration_range, label):
        attack_files = list(attack_audio_cache.keys())
        if not attack_files:
            return
        
        for normal_file, segments in tqdm(normal_segments_cache.items()):
            for normal_segment in segments:
                # 1. 随机选择攻击文件
                attack_file = random.choice(attack_files)
                attack_audio = attack_audio_cache[attack_file]
                
                # 2. 生成单个攻击持续时间
                attack_duration = random.randint(*attack_duration_range)
                
                # 3. 有效性检查
                if len(attack_audio) < attack_duration:
                    continue  # 跳过不满足条件的攻击音频
                    
                # 4. 随机选择单个叠加位置
                attack_start = random.randint(0, len(attack_audio) - attack_duration)
                pos_range = len(normal_segment) - attack_duration
                if pos_range <= 0:
                    continue  # 确保有可用位置
                    
                position = random.randint(0, pos_range)  # 单个随机位置
                
                # 5. 生成混合音频
                attack_segment = attack_audio[attack_start:attack_start+attack_duration]
                mixed = normal_segment.overlay(attack_segment, position=position)
                
                # 6. 提取特征并添加
                features = extract_features(mixed)
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

# WideResNet50特征提取器

# 特征保存逻辑
def save_features(features, labels, save_dir):
    """保存特征与标签到.npz文件（支持逐个样本处理后的合并）"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "features.npz")
    np.savez(
        save_path,
        features=features,
        labels=labels
    )

def compute_global_stats(X):
    """计算全局统计量"""
    all_values = np.concatenate([x.flatten() for x in X])
    global_mean = np.mean(all_values)
    global_std = np.std(all_values)
    np.save(GLOBAL_STATS_FILE, [global_mean, global_std])
    return global_mean, global_std

def normalize_with_global(X, global_mean, global_std):
    """应用全局归一化"""
    return [(x - global_mean) / (global_std + 1e-8) for x in X]

def save_visualization(matrix, filename, index, global_mean, global_std):
    """保存可视化结果"""
    os.makedirs(VISUAL_SAVE_DIR, exist_ok=True)
    
    # 保存灰度图
    plt.figure(figsize=(8,6), dpi=150)
    plt.imshow(matrix, cmap='gray', 
              vmin=global_mean-3*global_std,
              vmax=global_mean+3*global_std)
    plt.axis('off')
    plt.savefig(f"{VISUAL_SAVE_DIR}/{filename}_seg{index}_gray.png",
               bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 保存热力图
    plt.figure(figsize=(8,6), dpi=150)
    plt.imshow(matrix, cmap='jet', 
              norm=LogNorm(vmin=max(matrix.min(), 1e-6)))
    plt.colorbar()
    plt.axis('off')
    plt.savefig(f"{VISUAL_SAVE_DIR}/{filename}_seg{index}_heatmap.png",
               bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == "__main__":
    # 加载预处理数据
    if os.path.exists(CACHE_FILE):
        data_dict = np.load(CACHE_FILE, allow_pickle=True).item()
        X, y = data_dict['X'], data_dict['y']
    else:
        X, y = prepare_dataset('normal', 'attack')
    
    # 计算全局统计量（如果尚未计算）
    if os.path.exists(GLOBAL_STATS_FILE):
        global_mean, global_std = np.load(GLOBAL_STATS_FILE)
    else:
        global_mean, global_std = compute_global_stats(X)
    
    # 应用全局归一化
    X_normalized = normalize_with_global(X, global_mean, global_std)
    
    # 修改后的可视化保存代码
    print("保存可视化样本...")
    # 分别获取前10个正常和异常样本的索引
    normal_indices = np.where(y == 0)[:10]
    # print(normal_indices)
    attack_indices = np.where(y == 1)[:10]

    # 保存正常样本的可视化
    for idx in normal_indices[0][:10]:
        # print(idx)
        orig = X[idx]
        norm = X_normalized[idx]
        filename = f"normal_{idx}"
        # 保存原始特征图
        save_visualization(orig, filename+"_orig", 0, global_mean, global_std)
        # 保存归一化特征图
        save_visualization(norm, filename+"_norm", 0, global_mean, global_std)

    # 保存异常样本的可视化
    for idx in attack_indices[0][:10]:
        orig = X[idx]
        norm = X_normalized[idx]
        filename = f"attack_{idx}"
        # 保存原始特征图
        save_visualization(orig, filename+"_orig", 0, global_mean, global_std)
        # 保存归一化特征图
        save_visualization(norm, filename+"_norm", 0, global_mean, global_std)
        
        # 转换为numpy数组
        X_normalized = np.array(X_normalized)
    
    # 保存归一化后的特征
    save_features(X_normalized, y, FEATURE_SAVE_DIR)
    print(f"归一化后特征形状：{X_normalized.shape}")
    print(f"全局统计量 - 均值：{global_mean:.4f}, 标准差：{global_std:.4f}")