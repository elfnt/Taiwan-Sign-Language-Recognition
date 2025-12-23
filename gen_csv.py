import cv2
import mediapipe as mp
import os
import csv
import glob
import numpy as np

# --- 設定區 ---
ROOT_DIR = './output'  # 掃描 output 底下所有資料夾
OUTPUT_CSV = 'hand_landmarks_data2.csv'

# MediaPipe 設定（完全保持原樣）
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 定義 21 個手部關鍵點名稱 (參照 MediaPipe 官方文檔)
LANDMARK_NAMES = [
    'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]

def create_csv_header():
    header = ['video_name', 'frame_idx', 'label']
    for hand_type in ['Left', 'Right']:
        for lm_name in LANDMARK_NAMES:
            for axis in ['x', 'y', 'z']:
                header.append(f"{hand_type}_{lm_name}_{axis}")
    return header


def process_videos():
    # 準備寫入 CSV
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 寫入標頭
        header = create_csv_header()
        writer.writerow(header)

        # 遞迴搜索 output/ 內所有 mp4
        video_files = glob.glob(os.path.join(ROOT_DIR, "**/*.mp4"), recursive=True)
        print(f"找到 {len(video_files)} 個影片檔案，開始處理...")

        for video_path in video_files:
            filename = os.path.basename(video_path)

            # 保留你原本的 label 取得方式（檔名前段）
            try:
                label = filename.split('_')[0]
            except:
                label = "unknown"

            print(f"正在處理: {filename} (Label: {label})")

            cap = cv2.VideoCapture(video_path)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                left_hand_data = [0.0] * 63
                right_hand_data = [0.0] * 63

                if results.multi_hand_landmarks and results.multi_handedness:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        hand_label = results.multi_handedness[idx].classification[0].label

                        landmarks_flat = []
                        for lm in hand_landmarks.landmark:
                            landmarks_flat.extend([lm.x, lm.y, lm.z])

                        if hand_label == 'Left':
                            left_hand_data = landmarks_flat
                        elif hand_label == 'Right':
                            right_hand_data = landmarks_flat

                row = [filename, frame_idx, label] + left_hand_data + right_hand_data
                writer.writerow(row)

                frame_idx += 1

            cap.release()

    print(f"\n處理完成！資料已儲存至 {OUTPUT_CSV}")


if __name__ == "__main__":
    process_videos()
