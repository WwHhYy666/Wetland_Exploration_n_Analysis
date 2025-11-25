from ultralytics import YOLO
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体 (防止Matplotlib中文乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def analyze_wetland_vegetation(model_path, data_dir, output_dir):
    # 加载模型
    model = YOLO(model_path)
    
    # 获取类别名称
    class_names = model.names
    
    # 初始化统计数据
    stats_list = []
    
    # 获取图片
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg")) + \
                  glob.glob(os.path.join(data_dir, "*.png"))
    
    print(f"开始分析 {len(image_paths)} 张影像数据...")

    for img_path in image_paths:
        # 进行推理，不保存图片，只拿数据
        results = model.predict(img_path, verbose=False)
        result = results[0]
        
        img_h, img_w = result.orig_shape
        img_area = img_h * img_w
        
        # 统计单张图片中各类别的面积
        frame_stats = {name: 0 for name in class_names.values()}
        frame_stats['filename'] = os.path.basename(img_path)
        
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]
                
                # 获取框的坐标 (xyxy)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 计算矩形框面积
                box_area = (x2 - x1) * (y2 - y1)
                
                # 累加面积
                frame_stats[cls_name] += box_area

        # 计算百分比 (注意：如果是密集目标检测，框可能有重叠，总和可能超过100%，这在生态统计中叫“盖度”)
        for name in class_names.values():
            frame_stats[f"{name}_ratio"] = (frame_stats[name] / img_area) * 100
            
        stats_list.append(frame_stats)

    # 转换为 DataFrame
    df = pd.DataFrame(stats_list)
    
    # ================= 结果可视化 =================
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 保存原始数据 CSV
    csv_path = os.path.join(output_dir, "vegetation_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"统计数据已保存: {csv_path}")

    # 计算整个数据集的平均覆盖率
    ratio_cols = [col for col in df.columns if "_ratio" in col]
    avg_ratios = df[ratio_cols].mean()
    
    # 清理列名用于显示 (去掉 _ratio)
    labels = [label.replace("_ratio", "") for label in avg_ratios.index]
    
    # 2. 绘制总体占比饼图
    plt.figure(figsize=(10, 8))
    plt.pie(avg_ratios, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title("沉湖湿地航拍样本-植被平均覆盖度估算")
    plt.savefig(os.path.join(output_dir, "coverage_pie_chart.png"))
    plt.close()

    # 3. 绘制箱线图 (查看分布的离散程度)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[ratio_cols])
    plt.xticks(range(len(labels)), labels)
    plt.title("各航拍帧植被覆盖度分布范围")
    plt.ylabel("覆盖度 (%) - 基于检测框面积")
    plt.savefig(os.path.join(output_dir, "coverage_boxplot.png"))
    plt.close()
    
    print(f"✅ 分析完成！图表已保存至: {output_dir}")

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    
    # 配置
    # 指向你训练好的最佳模型
    MODEL_PATH = project_root / "runs" / "train" / "wetland_yolo11x_exp1" / "weights" / "best.pt"
    
    # 待分析的图片文件夹 (可以是之前的测试集，或者是新的航拍图)
    DATA_DIR = project_root / "data" / "wetland_dataset" / "images" / "val"
    
    # 结果保存
    OUTPUT_DIR = project_root / "results" / "statistics"
    
    if MODEL_PATH.exists() and DATA_DIR.exists():
        analyze_wetland_vegetation(MODEL_PATH, DATA_DIR, OUTPUT_DIR)
    else:
        print("错误: 找不到模型或数据文件夹。")