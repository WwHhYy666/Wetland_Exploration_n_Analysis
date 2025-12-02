import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict
import torch

# ================= 配置区域 =================
# 使用 r"" (raw string) 防止Windows路径中的反斜杠被转义
MODEL_PATH = r"E:\\Wetland_Exploration_n_Analysis\\models\\best.pt"
VIDEO_FOLDER = r"E:\\Wetland_Exploration_n_Analysis\\data\\self_dataset\\videos"
OUTPUT_FOLDER = r"E:\\Wetland_Exploration_n_Analysis\\runs\\analysis_results"

# 抽帧频率 (每30帧算一次，相当于每秒1次，如果想要更精细可以设为 10 或 5)
FRAME_INTERVAL = 30
# 置信度阈值
CONF_THRESHOLD = 0.5
# ===========================================

def batch_analyze_videos():
    # 0. 准备工作：检查设备和输出目录
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"当前运行设备: {device.upper()}")
    if device == 'cpu':
        print("警告: 使用CPU处理多个视频速度会较慢。")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 1. 加载模型
    print(f"正在加载模型: {MODEL_PATH} ...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"模型加载失败! 请检查路径。错误信息: {e}")
        return

    # 获取所有视频文件
    video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
    if not video_files:
        print(f"错误: 在 {VIDEO_FOLDER} 下未找到任何 .mp4 视频文件。")
        return
    
    print(f"共发现 {len(video_files)} 个视频文件，准备开始处理...")

    # 全局统计字典 (累加所有视频的结果)
    global_pixel_counts = defaultdict(int)
    class_names = model.names

    # 2. 循环处理每个视频
    for idx, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        print(f"\n[{idx+1}/{len(video_files)}] 正在处理视频: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  -> 无法打开视频，跳过。")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # 抽帧处理
            if frame_count % FRAME_INTERVAL != 0:
                continue
            
            # 推理 (使用 stream=True 可以减少内存占用，但这里单帧推影响不大)
            # retina_masks=True 保证mask质量
            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False, device=device, retina_masks=True)
            result = results[0]

            if result.masks is not None:
                # 提取数据
                masks = result.masks.data.cpu().numpy() # (N, H, W)
                classes = result.boxes.cls.cpu().numpy()
                
                # 统计当前帧各类别像素
                for i, class_id in enumerate(classes):
                    class_name = class_names[int(class_id)]
                    # 统计Mask中像素值为1的数量
                    pixel_sum = np.sum(masks[i])
                    global_pixel_counts[class_name] += pixel_sum
            
            processed_frames += 1
            if processed_frames % 10 == 0:
                print(f"  -> 进度: {frame_count}/{total_frames} 帧...", end='\r')

        cap.release()
        print(f"  -> {video_name} 处理完成。")

    # 3. 数据可视化 (生成总饼状图)
    print("\n所有视频处理完毕，正在生成统计图表...")
    
    if not global_pixel_counts:
        print("未检测到任何植被目标，无法生成图表。请检查置信度阈值或模型效果。")
        return

    labels = list(global_pixel_counts.keys())
    sizes = list(global_pixel_counts.values())
    
    # 排序：占比大的在前面
    sorted_pairs = sorted(zip(sizes, labels), reverse=True)
    sizes, labels = zip(*sorted_pairs)

    # 绘图
    plt.figure(figsize=(12, 10))
    # 颜色映射 (可选，让图表更好看)
    colors = plt.cm.Pastel1(np.arange(len(labels)))
    
    wedges, texts, autotexts = plt.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        shadow=True,
        colors=colors,
        pctdistance=0.85 # 百分比距离圆心的距离
    )
    
    # 使得饼图中间留白（变成甜甜圈图，看起来更科研一点）
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.axis('equal')  
    plt.title(f"Wetland Vegetation Distribution Analysis\n(Total {len(video_files)} Videos Aggregated)", fontsize=16)
    plt.legend(wedges, labels, title="Vegetation Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # 保存图片
    save_path = os.path.join(OUTPUT_FOLDER, 'total_vegetation_distribution.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"统计结果已保存至: {save_path}")
    
    # 同时打印文本报告
    print("\n====== 最终统计报告 ======")
    total_pixels = sum(sizes)
    for label, size in zip(labels, sizes):
        percentage = (size / total_pixels) * 100
        print(f"{label}: {percentage:.2f}%")
    print("==========================")
    plt.show()

if __name__ == '__main__':
    batch_analyze_videos()