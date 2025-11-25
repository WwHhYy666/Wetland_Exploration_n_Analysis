from ultralytics import YOLO
import os
import yaml
from pathlib import Path

def create_data_yaml(dataset_path, classes_list, yaml_path):
    """
    自动生成 dataset.yaml 配置文件
    """
    data_config = {
        'path': str(dataset_path.absolute()), # 数据集根目录
        'train': 'images/train',  # 训练图片相对路径
        'val': 'images/val',      # 验证图片相对路径
        'names': {i: name for i, name in enumerate(classes_list)} # 类别字典
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)
    print(f"✅ 已生成训练配置文件: {yaml_path}")

def train_yolo():
    # ================= 配置区域 =================
    # 1. 路径设置
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    
    # 数据集位置 (必须是上一步 split_dataset 生成的路径)
    dataset_dir = project_root / "data" / "wetland_dataset"
    yaml_path = dataset_dir / "wetland.yaml"
    
    # 2. 类别设置 (必须与你标注时的 classes.txt 顺序完全一致！！！)
    # 例如: 0: reed, 1: cattail...
    CLASS_NAMES = ["reed", "cattail", "water", "boat"] 
    
    # 3. 训练参数
    MODEL_WEIGHTS = "yolo11x.pt"  # 使用 yolov11x 预训练权重
    EPOCHS = 100                  # 训练轮数 (湿地场景复杂，建议100起)
    IMG_SIZE = 640                # 输入尺寸 (640是标准，1280适合高分辨率航拍但显存消耗巨大)
    BATCH_SIZE = 4                # ⚠️显存小请设为 2 或 4；显存大(>16G)可设为 16
    DEVICE = 0                    # GPU ID (0, 1...) 或 'cpu'
    
    # 结果保存路径
    project_dir = project_root / "runs" / "train"
    name_exp = "wetland_yolo11x_exp1"
    # ===========================================

    # 检查数据集是否存在
    if not dataset_dir.exists():
        print(f"❌ 错误: 找不到数据集 {dataset_dir}")
        print("请先运行 split_dataset.py")
        return

    # 1. 生成 yaml 配置文件
    create_data_yaml(dataset_dir, CLASS_NAMES, yaml_path)
    
    # 2. 加载模型
    print(f"正在加载模型: {MODEL_WEIGHTS} ...")
    try:
        model = YOLO(MODEL_WEIGHTS)
    except Exception as e:
        print("首次运行会自动下载权重文件，如果下载失败请检查网络。")
        raise e

    # 3. 开始训练
    print("🚀 开始训练... (按 Ctrl+C 可提前终止，模型会自动保存)")
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(project_dir), # 结果总目录
        name=name_exp,            # 本次实验名称
        pretrained=True,          # 加载预训练权重
        optimizer='auto',         # 自动选择优化器
        patience=20,              # 20轮不再提升则早停
        save=True,                # 保存 checkpoint
        exist_ok=True,            # 覆盖同名实验文件夹
        verbose=True
    )
    
    print("\n" + "="*40)
    print("✅ 训练完成！")
    print(f"最佳模型保存在: {project_dir}/{name_exp}/weights/best.pt")
    print("="*40)

if __name__ == "__main__":
    train_yolo()