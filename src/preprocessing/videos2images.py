import cv2
import os
import glob

def extract_frames_from_video(video_path, output_folder, interval=30):
    """
    从视频中按间隔抽取帧并保存为图片。
    
    Args:
        video_path (str): 视频文件的路径。
        output_folder (str): 图片保存的目标文件夹。
        interval (int): 抽帧间隔（每隔多少帧保存一张）。
    """
    # 获取视频文件名（不带后缀），用于给图片命名
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: 无法打开视频 {video_path}")
        return

    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"正在处理: {video_name} | FPS: {fps:.2f} | 总帧数: {total_frames}")

    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        # 如果读取不到帧（视频结束），跳出循环
        if not ret:
            break
        
        # 按照设定的间隔保存图片
        if frame_count % interval == 0:
            # 构造输出文件名: 视频名_帧序号.jpg
            image_name = f"{video_name}_frame_{frame_count:06d}.jpg"
            save_path = os.path.join(output_folder, image_name)
            
            # 保存图片
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"完成: {video_name} | 已保存图片: {saved_count} 张\n")

def batch_process(input_dir, output_dir, interval=30):
    """
    批量处理文件夹下的所有视频
    """
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 获取所有mp4文件
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    
    if not video_files:
        print(f"在 {input_dir} 中未找到 .mp4 文件。")
        return

    print(f"共发现 {len(video_files)} 个视频文件，准备开始处理...\n")

    for video_file in video_files:
        extract_frames_from_video(video_file, output_dir, interval)

    print("所有视频处理完毕！")

# ================= 配置区域 =================
if __name__ == "__main__":
    # 1. 设置视频所在的文件夹路径
    INPUT_VIDEO_DIR = r"./videos"  # 请修改为你的视频文件夹路径
    
    # 2. 设置图片保存的文件夹路径
    OUTPUT_IMAGE_DIR = r"./dataset_images" # 脚本会自动创建这个文件夹
    
    # 3. 设置抽帧间隔 (重要)
    # 无人机视频通常是30fps或60fps。
    # 设置为 30 表示每1秒存一张（如果是30fps视频）。
    # 如果视频变化很慢（如芦苇荡平飞），建议设置为 60 或 90（每2-3秒一张）。
    FRAME_INTERVAL = 60 

    batch_process(INPUT_VIDEO_DIR, OUTPUT_IMAGE_DIR, FRAME_INTERVAL)