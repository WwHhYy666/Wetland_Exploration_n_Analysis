import os
import glob

def check_progress(data_dir):
    # 获取所有图片
    extensions = ['*.jpg', '*.png', '*.jpeg']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(data_dir, ext)))
    
    # 获取所有txt标签
    txts = glob.glob(os.path.join(data_dir, "*.txt"))
    # 排除 classes.txt 和 predefined_classes.txt
    txts = [t for t in txts if "classes.txt" not in t]

    total_images = len(images)
    total_labels = len(txts)

    if total_images == 0:
        print("目录下没有图片。")
        return

    progress = (total_labels / total_images) * 100

    print("\n📊 标注进度报告")
    print("-" * 30)
    print(f"📂 目录: {data_dir}")
    print(f"🖼️  图片总数: {total_images}")
    print(f"🏷️  已标注数: {total_labels}")
    print(f"📈 完成度:   {progress:.2f}%")
    print("-" * 30)
    
    if total_labels > 0 and total_labels < total_images:
        remaining = total_images - total_labels
        print(f"💪 加油！还有 {remaining} 张图片等待标注。")
    elif total_images == total_labels:
        print("🎉 恭喜！所有图片已完成标注。")

if __name__ == "__main__":
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    TARGET_IMAGE_DIR = os.path.join(project_root, "data", "final_dataset_images")
    
    if os.path.exists(TARGET_IMAGE_DIR):
        check_progress(TARGET_IMAGE_DIR)