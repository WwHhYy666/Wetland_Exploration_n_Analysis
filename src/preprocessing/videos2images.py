import cv2
import os
import glob
from pathlib import Path

# ================= 修复后的配置区域 =================
# 获取当前脚本文件所在的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 拼接出 videos 和 images 的绝对路径
# 这样无论你在哪个目录下运行命令，脚本都能准确找到同级目录下的 videos
VIDEO_DIR = os.path.join(BASE_DIR, 'videos')
OUTPUT_DIR = os.path.join(BASE_DIR, 'images')

TIME_INTERVAL = 3.0 
JPEG_QUALITY = 95
# ===================================================

def extract_frames_from_video(video_path, output_folder, interval_sec):
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[Error] 无法打开视频: {video_path}")
        return
    # 获取视频的帧率 (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"[Error] 无法获取FPS，跳过: {video_path}")
        return

    frame_step = int(fps * interval_sec)
    if frame_step < 1: frame_step = 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"正在处理: {video_name} | FPS: {fps:.2f} | 总帧数: {total_frames}")

    current_frame = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if current_frame % frame_step == 0:
            out_name = f"{video_name}_{str(current_frame).zfill(6)}.jpg"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            saved_count += 1
        current_frame += 1

    cap.release()
    print(f"完成: {video_name} -> {saved_count} 张图片")

def main():
    # --- 调试信息打印 ---
    print(f"脚本所在路径: {BASE_DIR}")
    print(f"寻找视频路径: {VIDEO_DIR}")
    
    if not os.path.exists(VIDEO_DIR):
        print(f"\n[致命错误] 找不到 videos 文件夹！")
        print(f"请确认你的文件夹名字是否完全叫 'videos' (注意大小写)")
        print(f"当前目录下包含: {os.listdir(BASE_DIR)}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 支持更多无人机常见格式 (.MOV, .AVI, .mkv) 且忽略大小写
    exts = ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI']
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    
    video_files = sorted(list(set(video_files))) # 去重并排序

    if not video_files:
        print(f"\n[警告] 在 '{VIDEO_DIR}' 里面虽然有文件夹，但没找到视频文件。")
        print(f"请检查视频后缀名。该文件夹下的文件有: {os.listdir(VIDEO_DIR)}")
        return

    print(f"共发现 {len(video_files)} 个视频文件，开始处理...\n")

    for video_path in video_files:
        extract_frames_from_video(video_path, OUTPUT_DIR, TIME_INTERVAL)

    print("\n所有视频处理完毕！")
#
if __name__ == "__main__":
    main()