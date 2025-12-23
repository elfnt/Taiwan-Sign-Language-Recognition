import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import threading
from queue import Queue, Empty
import time
import joblib


# --- 優化配置 ---
OPTIMIZATION_CONFIG = {
    'SKIP_FRAMES': 1,           # 每 N 幀處理一次 MediaPipe
    'PREDICT_INTERVAL': 3,      # 每 N 幀執行一次模型預測 (降低預測頻率以提升 FPS)
    'RESIZE_FACTOR': 0.7,       # 影像縮放比例 (進一步縮小以加速)
    'DRAW_LANDMARKS': True,     # 是否繪製骨架
    'MODEL_COMPLEXITY': 1,      # MediaPipe 模型複雜度 (0=最快)
    'QUEUE_SIZE': 3,            # 執行緒間的佇列大小
}

# --- 1. 設定與初始化 ---
MODEL_PATH = 'tsl_fruit_model_v3.h5'
LABEL_ENCODER_PATH = 'label_encoder_v3.pkl'
SEQUENCE_LENGTH = 90
THRESHOLD = 0.8

# 全域控制變數
stop_threads = threading.Event()

# 載入模型與標籤
print("正在載入模型...")
model = load_model(MODEL_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
print("模型載入完成！可辨識類別:", le.classes_)

# MediaPipe 設定
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):
    """從 MediaPipe 結果中提取 126 維原始座標 (Left 63 + Right 63)"""
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

    return left_hand_data + right_hand_data


def normalize_frame(frame_data):
    """
    絕對座標轉相對座標 (以各手 wrist 為原點)，
    必須與錄影版 / 訓練時 normalize_frame 完全一致。
    """
    frame_data = np.array(frame_data, dtype=np.float32)
    left_hand = frame_data[:63]
    right_hand = frame_data[63:]

    # 左手
    if np.sum(np.abs(left_hand)) > 0:
        wrist = left_hand[:3]                      # 第一個關節 (WRIST)
        left_reshaped = left_hand.reshape(-1, 3)
        left_normalized = (left_reshaped - wrist).flatten()
    else:
        left_normalized = left_hand

    # 右手
    if np.sum(np.abs(right_hand)) > 0:
        wrist = right_hand[:3]
        right_reshaped = right_hand.reshape(-1, 3)
        right_normalized = (right_reshaped - wrist).flatten()
    else:
        right_normalized = right_hand

    return np.concatenate([left_normalized, right_normalized])


# --- 2. MediaPipe 處理執行緒 ---
def mediapipe_worker(input_queue, output_queue):
    """獨立執行緒：專門處理 MediaPipe 手部偵測"""
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=OPTIMIZATION_CONFIG['MODEL_COMPLEXITY'],
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    resize_factor = OPTIMIZATION_CONFIG['RESIZE_FACTOR']
    last_results = None

    try:
        while not stop_threads.is_set():
            try:
                image_data = input_queue.get(timeout=0.1)
                if image_data is None:
                    break

                frame_id, image, should_process = image_data

                if should_process:
                    if resize_factor < 1.0:
                        h, w = image.shape[:2]
                        new_w = int(w * resize_factor)
                        new_h = int(h * resize_factor)
                        small_image = cv2.resize(
                            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                        )
                        image_rgb = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
                    else:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    results = hands.process(image_rgb)
                    last_results = results
                else:
                    results = last_results

                output_queue.put((frame_id, results), timeout=0.1)

            except Empty:
                continue
            except Exception as e:
                print(f"MediaPipe 執行緒錯誤: {e}")
                break
    finally:
        hands.close()


# --- 3. 主迴圈 ---
def main():
    input_queue = Queue(maxsize=OPTIMIZATION_CONFIG['QUEUE_SIZE'])
    output_queue = Queue(maxsize=OPTIMIZATION_CONFIG['QUEUE_SIZE'])

    mp_thread = threading.Thread(
        target=mediapipe_worker,
        args=(input_queue, output_queue),
        daemon=True
    )
    mp_thread.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(0.5)

    sequence = []
    last_prediction_text = "..."
    last_confidence = 0.0
    last_results = None
    frame_count = 0

    # FPS 計算
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    # print("\n" + "=" * 60)
    # print("台灣手語辨識系統 - 多執行緒優化版 (含座標正規化)")
    # print("=" * 60)
    # print(f"✓ 多執行緒處理 (主迴圈 + MediaPipe 背景執行)")
    # print(f"✓ MediaPipe 跳幀: 每 {OPTIMIZATION_CONFIG['SKIP_FRAMES']} 幀處理一次")
    # print(f"✓ 模型預測跳幀: 每 {OPTIMIZATION_CONFIG['PREDICT_INTERVAL']} 幀預測一次")
    # print(f"✓ 影像縮放: {OPTIMIZATION_CONFIG['RESIZE_FACTOR']*100:.0f}%")
    # print("=" * 60)
    print("開始影像串流，按 'q' 離開...\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            fps_frame_count += 1

            # 計算 FPS (每 30 幀更新一次)
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                fps_frame_count = 0

            # 影像處理
            image = cv2.flip(frame, 1)

            # 決定這一幀是否需要處理 MediaPipe
            should_process = (frame_count % OPTIMIZATION_CONFIG['SKIP_FRAMES'] == 0)

            # 送入 MediaPipe 執行緒
            try:
                input_queue.put_nowait((frame_count, image, should_process))
            except:
                pass

            # 取得 MediaPipe 結果
            try:
                frame_id, results = output_queue.get(timeout=0.05)
                last_results = results
            except Empty:
                results = last_results

            # 繪製骨架
            if OPTIMIZATION_CONFIG['DRAW_LANDMARKS'] and results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )

            # 提取特徵 + 做座標正規化
            if results:
                raw_features = extract_keypoints(results)          # 126 維原始座標
                keypoints = normalize_frame(raw_features)          # 126 維相對座標
            else:
                # 如果沒有偵測到手，可以選擇清空序列，避免大量 0 污染
                keypoints = [0.0] * 126

            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            # 進行預測 (降低頻率以提升 FPS)
            if len(sequence) == SEQUENCE_LENGTH and frame_count % OPTIMIZATION_CONFIG['PREDICT_INTERVAL'] == 0:
                input_data = np.expand_dims(sequence, axis=0)  # (1, 90, 126)
                res = model.predict(input_data, verbose=0)[0]
                predicted_class_idx = np.argmax(res)
                confidence = res[predicted_class_idx]

                if confidence > THRESHOLD:
                    current_action = le.inverse_transform([predicted_class_idx])[0]
                    last_prediction_text = f"{current_action} ({confidence:.2f})"
                    last_confidence = confidence
                else:
                    last_prediction_text = "..."
                    last_confidence = 0.0

            # 顯示結果
            if len(sequence) == SEQUENCE_LENGTH:
                if last_confidence > THRESHOLD:
                    cv2.rectangle(image, (0, 0), (640, 80), (245, 117, 16), -1)
                    cv2.putText(image, last_prediction_text, (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255, 255, 255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(image, last_prediction_text, (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(image, f"Initializing... {len(sequence)}/{SEQUENCE_LENGTH}",
                            (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 255, 255), 3)

            # 顯示優化狀態
            cv2.imshow('Taiwan Sign Language Recognition - Optimized', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_threads.set()
        input_queue.put(None)
        mp_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()

        print("\n程式已結束")
        # print(f"總共處理了 {frame_count} 幀影像")


if __name__ == "__main__":
    main()