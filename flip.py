import cv2
import os
import glob

# 所有影片的主資料夾
ROOT_DIR = "./output"

def flip_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    # 取得影片參數
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 建立輸出影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 水平翻轉
        flipped = cv2.flip(frame, 1)

        out.write(flipped)

    cap.release()
    out.release()


def process_all_folders(root_dir):
    # 找所有分類資料夾（apple_3s、banana_3s…）
    folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        print(f"處理資料夾：{folder}")

        video_files = glob.glob(os.path.join(folder_path, "*.mp4"))

        for video_path in video_files:
            filename = os.path.basename(video_path)
            
            # 新名字：加上 _flip.mp4
            output_name = filename.replace(".mp4", "_flip.mp4")
            output_path = os.path.join(folder_path, output_name)

            # 已存在就跳過避免重複製作
            if os.path.exists(output_path):
                print(f"已存在跳過：{output_name}")
                continue

            print(f" 產生 → {output_name}")
            flip_video(video_path, output_path)

    print("\n🎉 全部影片水平翻轉完成！")


if __name__ == "__main__":
    process_all_folders(ROOT_DIR)
