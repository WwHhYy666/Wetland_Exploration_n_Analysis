from ultralytics import YOLO
import os
from pathlib import Path

def evaluate_model():
    # ================= 配置区域 =================
    # 1. 自动定位项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2] # src/test/script.py -> src -> test -> root
    
    # 2. 模型权重路径 (请根据你实际训练生成的路径修改)
    # 通常在 runs/train/你的实验名/weights/best.pt
    # 这里假设你上一步的实验名叫 wetland_yolo11x_exp1
    model_path = project_root / "runs" / "train" / "wetland_yolo11x_exp1" / "weights" / "best.pt"
    
    # 3. 数据集配置文件路径
    yaml_path = project_root / "data" / "wetland_dataset" / "wetland.yaml"
    
    # 4. 评估结果保存路径
    save_dir = project_root / "runs" / "val" / "final_evaluation"
    # ===========================================

    # 检查模型是否存在
    if not model_path.exists():
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        print("请检查路径，或确认训练是否已成功完成。")
        return

    print(f"正在加载模型: {model_path} ...")
    model = YOLO(model_path)

    print("\n🚀 开始在验证集上进行评估...")
    print("这可能需要几分钟，取决于验证集的大小...")

    # 运行验证模式
    # split='val' 表示使用验证集
    # save_json=True 可以保存原始数据方便后续自定义绘图
    # plots=True 确保生成 F1曲线、混淆矩阵等图片
    metrics = model.val(
        data=str(yaml_path),
        split='val',
        project=str(save_dir.parent),
        name=save_dir.name,
        plots=True,
        exist_ok=True
    )

    # ================= 打印核心指标 =================
    print("\n" + "="*40)
    print("📊 评估结果摘要 (Metrics Summary)")
    print("="*40)
    
    # map50: IoU阈值为0.5时的平均精度 (最常用的指标)
    print(f"mAP@0.5:      {metrics.box.map50:.4f} (越高越好)")
    
    # map50-95: IoU阈值从0.5到0.95的平均精度 (严苛指标)
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    
    # 打印每个类别的精确率(Precision)和召回率(Recall)
    # 注意: metrics.box.maps 包含每个类别的map
    print("-" * 40)
    print(f"{'类别':<15} {'mAP@0.5':<10}")
    print("-" * 40)
    
    # 获取类别名称
    names = model.names
    for i, ap in enumerate(metrics.box.maps):
        # maps 数组里通常包含了所有阈值的AP，这里简单展示
        # 如果要精确对应类别的AP50，ultralytics新版API稍微复杂一点
        # 这里直接读取 metrics.box.map50s (如果有的话) 或者直接看生成的 csv
        if i < len(names):
             print(f"{names[i]:<15} {metrics.box.maps[i]:.4f}")

    print("="*40)
    print(f"✅ 评估完成！")
    print(f"📈 所有的图表 (F1曲线, 混淆矩阵, PR曲线) 已保存至:")
    print(f"   📂 {save_dir}")
    print("="*40)

    # ================= 可视化预测测试 (可选) =================
    # 随机抽取几张验证集图片进行预测并保存，看看实际效果
    print("\n🖼️  正在生成可视化预测样例...")
    model.predict(
        source=str(project_root / "data" / "wetland_dataset" / "images" / "val"),
        max_det=20,     # 每张图最多检测20个目标
        conf=0.25,      # 置信度阈值
        save=True,      # 保存画框后的图片
        project=str(save_dir),
        name='visual_samples',
        exist_ok=True,
        max_det=50      # 限制每张图的检测数量
    )
    print(f"🖼️  样例图片已保存至: {save_dir}/visual_samples")

if __name__ == "__main__":
    evaluate_model()