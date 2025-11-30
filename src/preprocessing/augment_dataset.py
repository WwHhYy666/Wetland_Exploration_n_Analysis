import os
import shutil
from tqdm import tqdm

# === 配置区域 ===
# 你的数据集路径 (指向你 split 之后的 train 文件夹，或者清洗后的总文件夹)
# 建议先对总文件夹操作，然后再重新划分 train/val
SOURCE_IMAGES = "output_json2txt/output_images"
SOURCE_LABELS = "output_json2txt/yolotxt"

# 定义需要增强的类别 ID 和 目标扩充倍数
# 根据你的报告：
# 0: Phragmites (足够)
# 1: Miscanthus (154 -> 少) -> 扩充 3 倍
# 2: Typha (24 -> 极少) -> 扩充 15 倍 (救命级增强)
# 3: Nelumbo (足够)
# 4: Alternanthera (256 -> 偏少) -> 扩充 2 倍
# 5: Carex (足够)

# 注意：请根据你 CLASSES 列表的实际 ID 修改下面的键值对
# 假设你的 CLASSES = ["Phragmites", "Miscanthus", "Typha", "Nelumbo", "Alternanthera", "Carex"]
AUGMENT_RULES = {
    1: 3,   # Miscanthus
    2: 15,  # Typha (只有24个，必须疯狂复制)
    4: 2    # Alternanthera
}
# =================

def oversample():
    print("开始进行物理过采样...")
    txt_files = [f for f in os.listdir(SOURCE_LABELS) if f.endswith('.txt')]
    
    # 统计
    aug_count = 0
    
    for txt_file in tqdm(txt_files):
        txt_path = os.path.join(SOURCE_LABELS, txt_file)
        
        # 1. 检查该文件里有没有稀有类别
        has_rare_class = False
        max_multiplier = 0
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cls_id = int(line.split()[0])
                if cls_id in AUGMENT_RULES:
                    has_rare_class = True
                    # 如果一张图里同时有多个稀有类，取最大的倍数
                    max_multiplier = max(max_multiplier, AUGMENT_RULES[cls_id])
        
        # 2. 如果有，进行复制
        if has_rare_class:
            img_name = os.path.splitext(txt_file)[0]
            # 找图片后缀
            img_file = None
            for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                if os.path.exists(os.path.join(SOURCE_IMAGES, img_name + ext)):
                    img_file = img_name + ext
                    break
            
            if img_file:
                # 开始复制
                for i in range(max_multiplier):
                    new_name_base = f"{img_name}_augcopy_{i}"
                    
                    # 复制图片
                    src_img = os.path.join(SOURCE_IMAGES, img_file)
                    dst_img = os.path.join(SOURCE_IMAGES, new_name_base + os.path.splitext(img_file)[1])
                    shutil.copy2(src_img, dst_img)
                    
                    # 复制标签
                    dst_txt = os.path.join(SOURCE_LABELS, new_name_base + ".txt")
                    shutil.copy2(txt_path, dst_txt)
                    
                    aug_count += 1

    print(f"\n完成！共生成了 {aug_count} 份增强副本。")
    print("YOLO 在训练时会对这些副本进行随机旋转、裁剪，所以它们不会完全一样。")
    print("请重新运行 split_dataset.py 划分训练集和验证集。")

if __name__ == "__main__":
    oversample()