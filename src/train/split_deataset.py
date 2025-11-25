import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, target_dir, train_ratio=0.8):
    """
    将数据集划分为训练集和验证集，并按YOLO格式组织目录
    结构:
    dataset/
      images/
        train/
        val/
      labels/
        train/
        val/
    """
    # 定义目标路径
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs:
        os.makedirs(os.path.join(target_dir, d), exist_ok=True)

    # 获取所有图片文件 (假设标签文件名与图片一致，只是后缀不同)
    extensions = ['.jpg', '.png', '.jpeg']
    images = []
    for ext in extensions:
        images.extend(list(Path(source_dir).glob(f"*{ext}")))
    
    # 过滤掉没有对应txt标签的图片 (防止空数据报错)
    valid_pairs = []
    for img_path in images:
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
    
    if not valid_pairs:
        print("错误: 未找到匹配的图片和标签文件，请检查路径或是否已完成标注。")
        return

    # 随机打乱
    random.shuffle(valid_pairs)
    
    # 计算切分点
    split_idx = int(len(valid_pairs) * train_ratio)
    train_set = valid_pairs[:split_idx]
    val_set = valid_pairs[split_idx:]
    
    print(f"找到完整数据对: {len(valid_pairs)} 组")
    print(f"训练集: {len(train_set)} | 验证集: {len(val_set)}")
    print("正在复制文件，请稍候...")

    # 复制文件的辅助函数
    def copy_files(file_pairs, split_type):
        for img, lbl in file_pairs:
            # 复制图片
            shutil.copy2(img, os.path.join(target_dir, 'images', split_type, img.name))
            # 复制标签
            shutil.copy2(lbl, os.path.join(target_dir, 'labels', split_type, lbl.name))

    copy_files(train_set, 'train')
    copy_files(val_set, 'val')
    
    print(f"✅ 数据集划分完成！保存位置: {target_dir}")

if __name__ == "__main__":
    # 自动定位项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2] # src/tools/script.py -> src -> tools -> root
    
    # 源数据 (你标注好的文件夹)
    SOURCE_DATA = project_root / "data" / "final_dataset_images"
    
    # 目标数据 (YOLO训练专用文件夹)
    TARGET_DATA = project_root / "data" / "wetland_dataset"
    
    if not SOURCE_DATA.exists():
        print(f"源文件夹不存在: {SOURCE_DATA}")
    else:
        split_dataset(SOURCE_DATA, TARGET_DATA)