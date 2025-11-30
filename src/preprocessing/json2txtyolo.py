import os
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm  # 如果没有安装，请 pip install tqdm，或者删除相关代码

# ================= 项目配置区域 (请修改这里) =================

# 1. 你的类别名称 (必须与标注时的英文标签完全一致)
# 注意：YOLO的id是从0开始的，列表的顺序决定了id
CLASSES = ["Phragmites australis", "Miscanthus sacchariflorus", "Typha orientalis", "Nelumbo nucifera", "Alternanthera philoxeroides", "Carex spp."] 

# 2. 文件夹路径配置
JSON_FOLDER = "output_json"      # 原始JSON文件夹
IMAGE_FOLDER = "images"        # 原始图片文件夹

# 3. 输出路径配置
OUTPUT_ROOT = "output_json2txt"
OUTPUT_TXT_DIR = os.path.join(OUTPUT_ROOT, "yolotxt")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_ROOT, "output_images")

# ==========================================================

def setup_directories():
    """创建输出文件夹结构"""
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    
    # 如果子文件夹存在，建议先清空或直接覆盖，这里选择如果不存在则创建
    os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    print(f"文件夹已就绪: {OUTPUT_ROOT}")

def find_image_file(base_name, img_folder):
    """尝试寻找对应图片（支持 jpg, png, jpeg, bmp）"""
    extensions = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG", ".bmp"]
    for ext in extensions:
        img_path = os.path.join(img_folder, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None

def convert_single_file(json_file):
    """处理单个文件，返回是否成功及统计信息"""
    json_path = os.path.join(JSON_FOLDER, json_file)
    file_name_no_ext = os.path.splitext(json_file)[0]
    
    # 1. 读取 JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"\n[警告] JSON损坏无法读取: {json_file}")
        return False

    # 2. 寻找对应的图片
    src_img_path = find_image_file(file_name_no_ext, IMAGE_FOLDER)
    if not src_img_path:
        # 如果只有json没有图，跳过
        return False

    # 3. 获取图像宽高 (优先从JSON读，读不到再读图)
    h, w = data.get('imageHeight'), data.get('imageWidth')
    if not h or not w:
        img = cv2.imread(src_img_path)
        if img is None: return False
        h, w = img.shape[:2]

    # 4. 解析 Shapes
    yolo_lines = []
    shapes = data.get('shapes', [])
    
    if not shapes:
        return False # 空标签文件，跳过

    for shape in shapes:
        label = shape.get('label')
        points = shape.get('points')

        # 过滤：如果标签不在我们需要的目标列表里，或者没有点，跳过
        if label not in CLASSES or not points:
            continue
            
        class_id = CLASSES.index(label)
        np_points = np.array(points)
        
        # 归一化坐标并限制在 0-1 之间
        normalized_points = []
        for p in np_points:
            x = max(0, min(1, p[0] / w))
            y = max(0, min(1, p[1] / h))
            normalized_points.extend([x, y])
            
        # 只有构成多边形(至少3个点 -> 6个数值)才算有效
        if len(normalized_points) >= 6:
            line_str = f"{class_id} " + " ".join([f"{val:.6f}" for val in normalized_points])
            yolo_lines.append(line_str)

    # 5. 保存结果
    # 只有当 yolo_lines 不为空时，才保存 TXT 和 复制图片
    if len(yolo_lines) > 0:
        # 保存 TXT
        txt_filename = file_name_no_ext + ".txt"
        with open(os.path.join(OUTPUT_TXT_DIR, txt_filename), 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_lines))
        
        # 复制 图片
        dst_img_path = os.path.join(OUTPUT_IMG_DIR, os.path.basename(src_img_path))
        shutil.copy2(src_img_path, dst_img_path)
        
        return True
    else:
        # JSON存在但没有有效类别（比如全是不关心的杂草），跳过
        return False

def main():
    setup_directories()
    
    # 获取所有 json 文件
    json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith('.json')]
    total_files = len(json_files)
    
    print(f"检测到 {total_files} 个 JSON 文件，开始处理...")
    
    converted_count = 0
    skipped_count = 0
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(json_files)
    for json_file in pbar:
        success = convert_single_file(json_file)
        if success:
            converted_count += 1
        else:
            skipped_count += 1
        pbar.set_description(f"有效: {converted_count} | 跳过: {skipped_count}")

    print("\n" + "="*30)
    print(f"处理完成！")
    print(f"源文件总数: {total_files}")
    print(f"成功转换并保存: {converted_count}")
    print(f"被筛除(无标/空标/图片缺失): {skipped_count}")
    print(f"结果保存在: {os.path.abspath(OUTPUT_ROOT)}")
    print("="*30)

if __name__ == "__main__":
    main()