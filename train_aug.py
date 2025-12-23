import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- 設定區 ---
CSV_FILE = 'hand_landmarks_data.csv'
MODEL_SAVE_PATH = 'tsl_fruit_model_v3.h5'  # 升級為 v3
LABEL_ENCODER_PATH = 'label_encoder_v3.pkl'
SEQUENCE_LENGTH = 90
FEATURE_COUNT = 126
AUGMENT_FACTOR = 10  # 每個原始影片產生多少個變異版本 (建議 10-20)

def normalize_frame(frame_data):
    """ 座標相對化 (與 v2 相同) """
    frame_data = np.array(frame_data, dtype=np.float32)
    left_hand = frame_data[:63]
    right_hand = frame_data[63:]
    
    if np.sum(np.abs(left_hand)) > 0:
        wrist = left_hand[:3]
        left_reshaped = left_hand.reshape(-1, 3)
        left_normalized = (left_reshaped - wrist).flatten()
    else:
        left_normalized = left_hand

    if np.sum(np.abs(right_hand)) > 0:
        wrist = right_hand[:3]
        right_reshaped = right_hand.reshape(-1, 3)
        right_normalized = (right_reshaped - wrist).flatten()
    else:
        right_normalized = right_hand

    return np.concatenate([left_normalized, right_normalized])

def augment_sequence(sequence):
    """
    對單一序列進行資料增強
    input: (90, 126)
    output: List of (90, 126)
    """
    augmented_sequences = []
    
    # 1. 原始資料一定要保留
    augmented_sequences.append(sequence)
    
    # 產生 AUGMENT_FACTOR 個變異
    for _ in range(AUGMENT_FACTOR):
        new_seq = sequence.copy()
        
        # A. 隨機縮放 (Scaling): 模擬手的大小/遠近 (0.8x ~ 1.2x)
        scale_factor = np.random.uniform(0.85, 1.15)
        new_seq = new_seq * scale_factor
        
        # B. 隨機雜訊 (Jittering): 模擬座標抖動 (±0.02)
        noise = np.random.normal(0, 0.02, new_seq.shape)
        new_seq = new_seq + noise
        
        # C. 隨機時間位移 (Time Shift) - 簡單版
        # 隨機將序列向前或向後平移 1-5 幀，補 0
        shift = np.random.randint(-5, 5)
        if shift > 0: # 向後移 (前面補0)
            new_seq = np.vstack([np.zeros((shift, 126)), new_seq[:-shift]])
        elif shift < 0: # 向前移 (後面補0)
            new_seq = np.vstack([new_seq[abs(shift):], np.zeros((abs(shift), 126))])
            
        augmented_sequences.append(new_seq)
        
    return augmented_sequences

def load_and_process_data(csv_path):
    print("正在讀取並進行資料增強 (這可能會花一點時間)...")
    df = pd.read_csv(csv_path)
    
    # 標籤處理
    df['fruit_label'] = df['video_name'].apply(lambda x: x.split('_')[0])
    unique_labels = df['fruit_label'].unique()
    print(f"類別: {unique_labels}")

    feature_cols = [c for c in df.columns if 'WRIST' in c or 'FINGER' in c or 'THUMB' in c or 'PINKY' in c]
    
    videos = df['video_name'].unique()
    X = []
    y = []
    
    print(f"原始影片數: {len(videos)}")
    
    for video in videos:
        video_df = df[df['video_name'] == video].sort_values('frame_idx')
        features = video_df[feature_cols].values
        
        # 1. 正規化
        normalized_features = np.array([normalize_frame(f) for f in features])
        
        # 2. 補幀/截斷
        if len(normalized_features) >= SEQUENCE_LENGTH:
            base_sequence = normalized_features[:SEQUENCE_LENGTH]
        else:
            padding = np.zeros((SEQUENCE_LENGTH - len(normalized_features), FEATURE_COUNT))
            base_sequence = np.vstack((normalized_features, padding))
        
        # 3. 資料增強 (這是關鍵！)
        aug_seqs = augment_sequence(base_sequence)
        
        label = video_df['fruit_label'].iloc[0]
        
        # 將增強後的資料全部加入列表
        for seq in aug_seqs:
            X.append(seq)
            y.append(label)
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # 1. 資料處理
    X, y_str = load_and_process_data(CSV_FILE)
    print(f"增強後總樣本數: {len(X)}")

    # 2. 標籤編碼
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_str)
    y_categorical = to_categorical(y_encoded)
    CLASSES = list(le.classes_)
    joblib.dump(le, LABEL_ENCODER_PATH)

    # 3. 切分資料
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)

    # 4. 模型架構 (稍微輕量化以避免 overfitting)
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, FEATURE_COUNT)),
        
        Bidirectional(LSTM(64, return_sequences=True)), # 第一層保留 64
        Dropout(0.4), # 增加 Dropout 防止過擬合
        
        Bidirectional(LSTM(32, return_sequences=False)), # 第二層縮減為 32
        Dropout(0.4),
        
        BatchNormalization(),
        Dense(32, activation='relu'), # Dense 層也縮減
        Dropout(0.4),
        
        Dense(len(CLASSES), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 5. Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True), # 耐心增加
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # 6. 開始訓練
    print("開始訓練 V3 模型...")
    history = model.fit(
        X_train, y_train, 
        epochs=150, # 增加 epochs
        batch_size=32, # 稍微加大 batch
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    print(f"模型訓練完成，已儲存至 {MODEL_SAVE_PATH}")
    
    # 簡單繪圖
    try:
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Val')
        plt.title('Model Accuracy (Augmented)')
        plt.legend()
        plt.savefig('training_history_v3.png')
        print("訓練曲線已儲存為 training_history_v3.png")
    except:
        pass