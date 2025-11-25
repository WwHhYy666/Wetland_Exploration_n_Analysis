import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
import random
from pathlib import Path

def generate_vegetation_map(model_path, input_dir, output_dir):
    model = YOLO(model_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 为每个类别定义一种颜色 (BGR格式)
    # 芦苇: 绿色, 香蒲: 黄色, 水面: 蓝色, 船: 红色
    # 你可以根据 classes.txt 的顺序调整这里
    colors = [
        (0, 255, 0),    # Class 0: Green (Reed)
        (0, 255, 255),  # Class 1: Yellow (Cattail)
        (255, 0, 0),    # Class 2: Blue (Water)
        (0, 0, 255)     # Class 3: Red (Boat)
    ]
    
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    # 为了演示，只取前20张，实际使用可以去掉切片
    process_paths = image_paths[:20] 
    
    print(f"正在生成植被结构图，共 {len(process_paths)} 张...")

    for img_path in process_paths:
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 创建一个覆盖层 (Overlay) 用于画框
        overlay = img.copy()
        
        results = model.predict(img, verbose=False)[0]
        
        if results.boxes:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 获取该类别的颜色，如果超出定义则随机
                color = colors[cls_id] if cls_id < len(colors) else (128, 128, 128)
                
                # 在覆盖层上画实心矩形
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                # 可选：画个边框让轮廓更清晰
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 核心步骤：将覆盖层与原图混合
        # alpha=0.4 表示覆盖层只有40%不透明度，这样可以看到底下的纹理，
        # 同时框重叠的地方颜色会变深，体现"密度"。
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # 添加图例 (Legend) - 简单写在左上角
        y_offset = 30
        for i, name in enumerate(results.names.values()):
            c = colors[i] if i < len(colors) else (128, 128, 128)
            cv2.rectangle(img, (10, y_offset - 20), (30, y_offset), c, -1)
            cv2.putText(img, name, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        # 保存
        save_name = "structure_" + os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, save_name), img)

    print(f"✅ 植被结构图生成完毕: {output_dir}")

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    
    MODEL_PATH = project_root / "runs" / "train" / "wetland_yolo11x_exp1" / "weights" / "best.pt"
    INPUT_DIR = project_root / "data" / "wetland_dataset" / "images" / "val"
    OUTPUT_DIR = project_root / "results" / "structure_maps"
    
    if MODEL_PATH.exists():
        generate_vegetation_map(MODEL_PATH, INPUT_DIR, OUTPUT_DIR)