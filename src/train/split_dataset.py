import os
import shutil
import random
from tqdm import tqdm

# ================= 配置 =================
# 1. 你刚才生成的清洗后的数据文件夹
SOURCE_ROOT = "output_json2txt"
SRC_IMAGES = os.path.join(SOURCE_ROOT, "output_images")
SRC_LABELS = os.path.join(SOURCE_ROOT, "yolotxt")

# 2. 准备要把数据存到哪里去 (YOLO会自动读取这个位置)
TARGET_ROOT = "datasets/chenhu_seg"

# 3. 划分比例 (0.2 表示 20% 做验证集)
VAL_RATIO = 0.2
# =======================================

def split_data():
    # 检查源文件是否存在
    if not os.path.exists(SRC_IMAGES):
        print("找不到图片文件夹，请检查路径")
        return

    # 创建目标文件夹结构
    for split in ['train', 'val']:
        os.makedirs(os.path.join(TARGET_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(TARGET_ROOT, split, 'labels'), exist_ok=True)

    # 获取所有图片文件
    images = [f for f in os.listdir(SRC_IMAGES) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    # 打乱顺序，保证随机性
    random.seed(42)
    random.shuffle(images)
    
    # 计算划分数量
    val_count = int(len(images) * VAL_RATIO)
    train_count = len(images) - val_count
    
    print(f"总数据: {len(images)} 张")
    print(f"训练集: {train_count} 张 | 验证集: {val_count} 张")
    print(f"目标目录: {TARGET_ROOT}")

    # 开始移动/复制文件
    for i, img_name in enumerate(tqdm(images)):
        # 决定是 train 还是 val
        if i < val_count:
            split = "val"
        else:
            split = "train"
            
        # 构造源路径和目标路径
        src_img_path = os.path.join(SRC_IMAGES, img_name)
        dst_img_path = os.path.join(TARGET_ROOT, split, 'images', img_name)
        
        # 处理对应的 txt
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        src_txt_path = os.path.join(SRC_LABELS, txt_name)
        dst_txt_path = os.path.join(TARGET_ROOT, split, 'labels', txt_name)
        
        # 复制文件
        shutil.copy2(src_img_path, dst_img_path)
        if os.path.exists(src_txt_path):
            shutil.copy2(src_txt_path, dst_txt_path)
        else:
            print(f"警告: 找不到对应的标签文件 {txt_name}")

    print("数据集划分完成！")

if __name__ == "__main__":
    split_data()