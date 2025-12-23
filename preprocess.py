import os
import subprocess
import cv2
import urllib.request
import zipfile
import shutil

# === 1. 設定工作路徑 (E槽) ===
ROOT_DIR = r"E:\ML_data"
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_FOLDER_NAME = "ffmpeg_tool" 

# 輸出資料夾根目錄
OUTPUT_ROOT = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# === 自動下載與設定 FFmpeg ===
def setup_ffmpeg():
    tool_dir = os.path.join(ROOT_DIR, FFMPEG_FOLDER_NAME)
    bin_dir = os.path.join(tool_dir, "bin")
    exe_path = os.path.join(bin_dir, "ffmpeg.exe")

    if os.path.exists(exe_path):
        return bin_dir

    print(f"⚠️ 偵測到缺少 FFmpeg，準備下載安裝...")
    print(f"⬇️ 下載中 (約 120MB)...")
    
    zip_path = os.path.join(ROOT_DIR, "ffmpeg.zip")
    try:
        urllib.request.urlretrieve(FFMPEG_URL, zip_path)
        print("📦 解壓縮中...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ROOT_DIR)
        
        extracted_folders = [f for f in os.listdir(ROOT_DIR) if "ffmpeg" in f and os.path.isdir(os.path.join(ROOT_DIR, f)) and f != "output" and f != FFMPEG_FOLDER_NAME]
        
        if not extracted_folders:
            raise RuntimeError("解壓縮失敗")

        original_folder = os.path.join(ROOT_DIR, extracted_folders[0])
        if os.path.exists(tool_dir):
            shutil.rmtree(tool_dir)
        os.rename(original_folder, tool_dir)
        os.remove(zip_path)
        print(f"✅ FFmpeg 安裝完成")
        return bin_dir
    except Exception as e:
        print(f"❌ 安裝失敗：{e}")
        exit()

# 設定環境變數
FFMPEG_BIN_PATH = setup_ffmpeg()
os.environ["PATH"] += os.pathsep + FFMPEG_BIN_PATH

EXTS = (".mp4", ".mov", ".MOV", ".avi", ".mkv")
IGNORE_FOLDERS = {"ffmpeg_tool", "output", "System Volume Information", "$RECYCLE.BIN"}

def get_video_duration(input_path):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): return 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0
    except: return 0

def process_video_ffmpeg(input_path, output_path):
    target_duration = 3.0
    target_w, target_h = 640, 480
    duration = get_video_duration(input_path)
    
    if duration <= 0:
        print(f"⚠️ 無法讀取：{os.path.basename(input_path)}")
        return

    cmd = ["ffmpeg", "-y"]
    
    # === 裁切邏輯判斷 (依據原始檔名) ===
    filename_only = os.path.basename(input_path)
    
    # 預設：置中裁切
    crop_cmd = f"crop={target_w}:{target_h}"
    
    # 特殊規則：原始檔名含 "9" -> 左下裁切
    if "9" in filename_only:
        crop_cmd = f"crop={target_w}:{target_h}:0:in_h-{target_h}"

    filter_chain = [
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase",
        crop_cmd,
        "setsar=1",
        "fps=30"
    ]

    # 時間處理
    if duration > target_duration:
        start_time = (duration - target_duration) / 2
        cmd.extend(["-ss", f"{start_time:.2f}"])
        cmd.extend(["-i", input_path])
    else:
        cmd.extend(["-i", input_path])
        filter_chain.append("tpad=stop_mode=clone:stop_duration=3")

    cmd.extend(["-vf", ",".join(filter_chain)])
    cmd.extend([
        "-t", str(target_duration),
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ])

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    except subprocess.CalledProcessError:
        print(f"❌ 失敗：{os.path.basename(input_path)}")

if __name__ == "__main__":
    print(f"🚀 開始處理")
    print(f"⚙️  命名規則：分類_分類編號 (e.g. apple_apple1_3s.mp4)")
    
    for folder_name in os.listdir(ROOT_DIR):
        folder_path = os.path.join(ROOT_DIR, folder_name)

        if os.path.isdir(folder_path) and folder_name not in IGNORE_FOLDERS and "ffmpeg" not in folder_name:
            
            sub_output_dir = os.path.join(OUTPUT_ROOT, f"{folder_name}_3s")
            os.makedirs(sub_output_dir, exist_ok=True)
            
            # 取得該資料夾下所有影片並排序
            files = sorted([f for f in os.listdir(folder_path) if f.endswith(EXTS)])
            
            if not files:
                continue
            
            print(f"\n📂 分類：{folder_name} (共 {len(files)} 個檔案)")

            for i, fname in enumerate(files, start=1):
                in_path = os.path.join(folder_path, fname)
                
                # === 關鍵修改：檔名格式 ===
                # 格式：分類名 + "_" + 分類名 + 數字 + "_3s.mp4"
                # 例如：apple_apple1_3s.mp4
                out_name = f"{folder_name}_{folder_name}{i}_3s.mp4"
                
                out_path = os.path.join(sub_output_dir, out_name)
                
                process_video_ffmpeg(in_path, out_path)

    print("\n🎉 全部完成！")