import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 必须与你转换时的类别顺序完全一致！
CLASSES = ["Phragmites australis", "Miscanthus sacchariflorus", "Typha orientalis", "Nelumbo nucifera", "Alternanthera philoxeroides", "Carex spp."] 

# 2. 路径配置 (指向你刚刚生成的文件夹)
INPUT_ROOT = "output_json2txt"
TXT_DIR = os.path.join(INPUT_ROOT, "yolotxt")
IMG_DIR = os.path.join(INPUT_ROOT, "output_images")

# 3. 验证结果保存位置
SAVE_DIR = "visualization_check"
# ===========================================

def generate_colors(num_classes):
    """为每个类别生成固定的随机颜色"""
    np.random.seed(42) # 固定随机种子，保证每次颜色一样
    colors = []
    for _ in range(num_classes):
        # 生成 BGR 颜色
        color = np.random.randint(0, 255, size=3).tolist()
        colors.append(color)
    return colors

def visualize():
    # 创建输出文件夹
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # 获取颜色板
    colors = generate_colors(len(CLASSES))

    # 获取所有txt文件
    txt_files = [f for f in os.listdir(TXT_DIR) if f.endswith('.txt')]
    
    # 为了快速验证，如果数据量太大，只抽取前20张看一看
    # 如果想看全部，注释掉下面这两行
    if len(txt_files) > 20:
        print(f"数据较多({len(txt_files)}张)，随机抽取20张进行验证...")
        txt_files = random.sample(txt_files, 20)

    print(f"开始生成可视化图片到 '{SAVE_DIR}' 文件夹...")

    for txt_file in tqdm(txt_files):
        # 1. 找到对应的图片路径
        basename = os.path.splitext(txt_file)[0]
        
        # 尝试寻找对应图片后缀
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            temp_path = os.path.join(IMG_DIR, basename + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        
        if img_path is None:
            print(f"[错误] 找不到图片: {basename}")
            continue

        # 2. 读取图片
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # 创建一个用于画半透明遮罩的图层
        overlay = img.copy()

        # 3. 读取并解析 TXT
        with open(os.path.join(TXT_DIR, txt_file), 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3: continue

            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            # 4. 反归一化坐标 (x * w, y * h)
            # YOLO格式: class x1 y1 x2 y2 ... xn yn
            points = []
            for i in range(0, len(coords), 2):
                pt_x = int(coords[i] * w)
                pt_y = int(coords[i+1] * h)
                points.append([pt_x, pt_y])
            
            pts_np = np.array(points, np.int32)
            pts_np = pts_np.reshape((-1, 1, 2))

            # 5. 绘制
            color = colors[class_id] if class_id < len(colors) else [255, 255, 255]
            
            # 填充多边形 (半透明效果)
            cv2.fillPoly(overlay, [pts_np], color)
            # 画轮廓 (实线)
            cv2.polylines(img, [pts_np], True, color, 2)
            
            # 写类别名称
            text_pos = (points[0][0], points[0][1] - 5)
            cv2.putText(img, CLASSES[class_id], text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 2)

        # 6. 混合图片 (原图 + 半透明遮罩)
        alpha = 0.4  # 透明度
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 7. 保存
        cv2.imwrite(os.path.join(SAVE_DIR, basename + "_vis.jpg"), img)

    print("验证完成！请打开文件夹查看图片是否正确。")

if __name__ == "__main__":
    visualize()