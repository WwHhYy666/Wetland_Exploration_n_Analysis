import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# === 配置 ===
LABEL_DIR = "output_json2txt/yolotxt"  # 你的txt文件夹路径
CLASSES = ["Phragmites australis", "Miscanthus sacchariflorus", "Typha orientalis", "Nelumbo nucifera", "Alternanthera philoxeroides", "Carex spp."] 
# ===========

def analyze():
    class_counts = {name: 0 for name in CLASSES}
    box_sizes = [] # 存储所有框的面积 (w * h)

    txt_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    
    print("正在分析标签分布...")
    for txt in tqdm(txt_files):
        with open(os.path.join(LABEL_DIR, txt), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue # 格式不对
                
                # 统计类别
                cls_id = int(parts[0])
                if cls_id < len(CLASSES):
                    class_counts[CLASSES[cls_id]] += 1
                
                # 统计目标大小 (YOLO分割格式: class x1 y1 x2 y2 ...)
                # 粗略计算外接矩形宽高
                coords = [float(x) for x in parts[1:]]
                xs = coords[0::2]
                ys = coords[1::2]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                box_sizes.append(w * h)

    # === 绘图 ===
    plt.figure(figsize=(12, 5))

    # 图1: 类别数量分布
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title("Class Distribution (Number of Instances)")
    plt.ylabel("Count")
    
    # 图2: 目标尺寸分布 (小目标检测难)
    plt.subplot(1, 2, 2)
    sns.histplot(box_sizes, bins=50, kde=True)
    plt.title("Object Size Distribution (Area ratio)")
    plt.xlabel("Area (Normalized 0~1)")
    
    plt.tight_layout()
    plt.savefig("dataset_analysis.png")
    print("分析完成！请查看 dataset_analysis.png")
    
    # === 打印诊断建议 ===
    print("\n=== 诊断报告 ===")
    total = sum(class_counts.values())
    for name, count in class_counts.items():
        ratio = count / total if total > 0 else 0
        print(f"{name}: {count} ({ratio:.1%})")
        if ratio < 0.1:
            print(f"  [警告] {name} 样本过少 (<10%)，可能导致该类别检测效果极差！建议复制粘贴增强数据。")
            
    small_obj = sum(1 for s in box_sizes if s < 0.01) # 面积小于全图1%的算小目标
    print(f"\n小目标数量: {small_obj} (占比 {small_obj/len(box_sizes):.1%})")
    if small_obj/len(box_sizes) > 0.5:
        print("  [警告] 超过50%的目标是非常小的物体。")
        print("  -> 必须使用 imgsz=1024 或 1280 训练，否则根本看不见。")

if __name__ == "__main__":
    analyze()