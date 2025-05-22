import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import wide_resnet50_2
from scipy.stats import entropy
import torch.nn.functional as F
from pydub import AudioSegment
import random
from tqdm import tqdm

# 1. 参数设置
SEGMENT_LENGTH = 0.3  # 300ms
SAMPLING_RATE = 48000  # 48kHz采样率
DEBUG_SPLIT = 0
CACHE_FILE = "./data.npy"
SAVE_PATH = "audio_classifier_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 特征提取函数（二维频谱混沌图）
def extract_spectral_entropy(audio, sr, patch_size=5, stride=2):
    """计算频谱混沌图（论文4.2节）"""
    n_fft = 512
    hop_length = 137
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
            entropy = -np.sum(power * np.log(power + 1e-10))
            E_p[i, j] = entropy
    
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
    
    return features  # 返回一个列表，每个元素是一个二维数组

# 3. 数据准备
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
    
    # 转换为numpy数组并保存
    X = np.array(X)  # 形状为 (样本数, rows_ep, cols_ep)
    y = np.array(y)
    data_dict = {'X': X, 'y': y}
    np.save(CACHE_FILE, data_dict)
    return X, y
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        print(x.shape)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        print(x.shape)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        print(x.shape)
        x = self.global_pool(x)
        print(x.shape)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # 加载数据
    if os.path.exists(CACHE_FILE):
        data_dict = np.load(CACHE_FILE, allow_pickle=True).item()
        X, y = data_dict['X'], data_dict['y']
    else:
        X, y = prepare_dataset('normal', 'attack')
    
     # 添加通道维度并转换为float32
    X = np.expand_dims(X, axis=1).astype(np.float32)  # 形状变为 (n_samples, 1, H, W)
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train).float().to(DEVICE)
    X_test_tensor = torch.tensor(X_test).float().to(DEVICE)
    y_train_tensor = torch.tensor(y_train).long().to(DEVICE)
    y_test_tensor = torch.tensor(y_test).long().to(DEVICE)
    
    # 创建数据集和DataLoader
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # 训练循环
    best_acc = 0.0
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Val Acc: {val_acc:.2f}% | Loss: {running_loss/len(train_loader):.4f}")
        scheduler.step(running_loss)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

    # 加载最佳模型进行评估
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Attack']))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()





# Base  Debug_split=100
# Classification Report:
#               precision    recall  f1-score   support

#       Normal       0.85      0.96      0.90      2533
#       Attack       0.95      0.82      0.88      2420

#     accuracy                           0.89      4953
#    macro avg       0.90      0.89      0.89      4953
# weighted avg       0.90      0.89      0.89      4953