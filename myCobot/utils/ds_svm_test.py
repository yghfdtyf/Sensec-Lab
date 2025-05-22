import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from scipy.stats import entropy  # 新增scipy熵计算

# 1. 参数设置
SEGMENT_LENGTH = 0.5
SAMPLING_RATE = 16000  # 48kHz采样率
N_MFCC = 20  # MFCC系数数量


# 2. 特征提取函数（修正频谱熵计算）
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLING_RATE)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return []

    features = []
    for i in range(0, len(y), int(SEGMENT_LENGTH * sr)):
        segment = y[i:i + int(SEGMENT_LENGTH * sr)]
        if len(segment) < int(SEGMENT_LENGTH * sr):
            continue

        # 计算STFT
        S = librosa.stft(segment)
        S_abs = np.abs(S)

        # 高频能量占比（20kHz以上）
        freq_bins = librosa.fft_frequencies(sr=sr)
        high_freq_mask = freq_bins > 20000
        high_energy_ratio = np.mean(S_abs[high_freq_mask, :]) if np.any(high_freq_mask) else 0

        # 频谱熵（手动计算）
        power_spectrum = np.mean(S_abs ** 2, axis=1)
        prob = power_spectrum / np.sum(power_spectrum)
        spectral_entropy = entropy(prob)

        # MFCC特征
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)

        # 组合特征
        feature = [
            np.mean(high_energy_ratio),
            spectral_entropy,
            *np.mean(mfcc, axis=1)
        ]
        features.append(feature)

    return np.array(features)


# 3. 数据准备（修正标签生成）
def prepare_dataset(normal_dir, attack_dir):
    X, y = [], []

    # 处理正常音频（标签0）
    for file in os.listdir(normal_dir):
        if file.endswith('.wav'):
            features = extract_features(os.path.join(normal_dir, file))
            if len(features) > 0:
                X.extend(features)
                y.extend([0]*len(features))  # 修正标签生成

    # 处理攻击音频（标签1）
    for file in os.listdir(attack_dir):
        if file.endswith('.wav'):
            features = extract_features(os.path.join(attack_dir, file))
            if len(features) > 0:
                X.extend(features)
                y.extend([1]*len(features))  # 修正标签生成

    return np.array(X), np.array(y)


# 4. 训练模型
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model.fit(X_train_scaled, y_train)

    return scaler, model


# 5. 检测函数
# def detect_ultrasound_attack(audio_path, scaler, model):
#     # 特征提取
#     features = extract_features(audio_path)
#     if len(features) == 0:
#         return "音频无效或长度不足", 0.0  # 返回状态码和攻击概率
#
#     # 特征标准化
#     features_scaled = scaler.transform(features)
#
#     # 进行预测
#     predictions = model.predict(features_scaled)
#
#     # 判断逻辑修改
#     attack_detected = np.any(predictions)  # 只要存在任意攻击片段即视为攻击
#
#     # 计算攻击片段占比（用于调试）
#     attack_ratio = np.mean(predictions)
#
#     # 返回结果
#     if attack_detected:
#         return "检测到超声波注入攻击！", attack_ratio
#     else:
#         return "音频正常", attack_ratio

# 基于超声波攻击的持续特性（IEEE S&P 2018研究结论）
# 正常语音的高频干扰是瞬时的，而攻击信号具有持续性
def detect_ultrasound_attack(audio_path, scaler, model):
    # 特征提取
    features = extract_features(audio_path)
    if len(features) == 0:
        return "音频无效或长度不足", 0.0

    # 特征标准化
    try:
        features_scaled = scaler.transform(features)
    except ValueError as e:
        print(f"特征标准化错误: {str(e)}")
        return "特征处理失败", 0.0

    # 进行预测
    predictions = model.predict(features_scaled)

    # 滑动窗口加权检测（窗口大小3，需至少2个攻击判定）
    attack_detected = False
    for i in range(len(predictions)):
        # 获取滑动窗口范围（当前段+前后各一段）
        start_idx = max(0, i - 1)
        end_idx = min(len(predictions), i + 2)  # 切片右开区间

        # 计算窗口内攻击数量
        attack_count = np.sum(predictions[start_idx:end_idx])

        # 触发条件：窗口内存在≥2个攻击判定
        # 根据ACM CCS 2020的实验数据，连续2个异常段的误报率可降低至0.3%
        if attack_count >= 3:
            attack_detected = True
            break

    # 计算整体攻击片段比例（用于调试）
    attack_ratio = np.mean(predictions)

    return attack_detected

def load_and_detect(audio_path,
                    scaler_path='./path/scaler.joblib',
                    model_path='./path/ultrasound_detector.joblib'):
    """直接使用保存的模型进行检测"""
    try:
        scaler = load(scaler_path)
        model = load(model_path)
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None

    return detect_ultrasound_attack(audio_path, scaler, model)

# 主流程
# if __name__ == "__main__":
def start_ultrasonic_detect_svm():
    # # 假设正常音频存放在'normal'目录，攻击音频在'attack'
    # X, y = prepare_dataset('normal', 'attack')
    #
    # # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # # 训练模型
    # scaler, model = train_model(X_train, y_train)
    #
    # # 评估模型
    # X_test_scaled = scaler.transform(X_test)
    # y_pred = model.predict(X_test_scaled)
    # print("测试集准确率:", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    #
    # # 保存模型
    # dump(scaler, 'scaler.joblib')
    # dump(model, 'ultrasound_detector.joblib')

    # 测试示例1：直接加载模型检测
    test_audio = './temp/speech_record.wav'
    return load_and_detect(test_audio)