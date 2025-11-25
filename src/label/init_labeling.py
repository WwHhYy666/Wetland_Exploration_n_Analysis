import os

def initialize_labeling_env(data_dir, classes):
    """
    初始化标注环境：生成 classes.txt 和 predefined_classes.txt
    
    Args:
        data_dir (str): 图片数据集所在的文件夹路径
        classes (list): 类别名称列表 (英文)
    """
    
    # 1. 生成 classes.txt (YOLO 训练必须文件)
    # 这个文件必须和图片/标签在同一目录，或者被 labelImg 读取
    classes_txt_path = os.path.join(data_dir, "classes.txt")
    
    print(f"正在生成类别文件...")
    with open(classes_txt_path, 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    print(f"✅ 已生成 YOLO 类别定义: {classes_txt_path}")
    print(f"   包含类别: {classes}")

    # 2. 生成 LabelImg 专用预设文件 (predefined_classes.txt)
    # 这让 LabelImg 启动时侧边栏直接就有这些选项
    predefined_path = os.path.join(data_dir, "predefined_classes.txt")
    with open(predefined_path, 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    print(f"✅ 已生成 LabelImg 预设列表: {predefined_path}")

    # 3. 生成启动命令建议
    print("\n" + "="*40)
    print("🚀 推荐的 LabelImg 启动命令：")
    print("="*40)
    print(f"labelImg {data_dir} {predefined_path}")
    print("="*40)
    print("提示：")
    print("1. 启动后，请按 'W' 键开始画框。")
    print("2. 确保左侧工具栏模式已切换为 [YOLO] (默认可能是 PascalVOC)。")
    print("3. 按 'D' 切换下一张，'A' 切换上一张。")

# ================= 配置区域 =================
if __name__ == "__main__":
    # 自动获取项目路径
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    
    # 指向你准备好要标注的图片文件夹 (例如之前的 final_dataset_images)
    # 注意：为了方便 LabelImg，建议把 txt 标签直接保存在图片同级目录下
    TARGET_IMAGE_DIR = os.path.join(project_root, "data", "final_dataset_images")
    
    # 定义你的植物类别 (请根据沉湖湿地实际情况修改)
    # 注意：YOLO 的类别 ID 是根据这个列表的顺序生成的 (0, 1, 2...)
    # 以后训练时顺序绝对不能变！！
    MY_CLASSES = [
        "reed",       # 芦苇 (ID: 0)
        "cattail",    # 香蒲 (ID: 1, 假设有)
        "water",      # 水面 (ID: 2, 如果需要作为负样本或者特定分割)
        "boat"        # 船只 (ID: 3, 干扰物等)
    ]

    if not os.path.exists(TARGET_IMAGE_DIR):
        print(f"❌ 错误: 找不到文件夹 {TARGET_IMAGE_DIR}")
        print("请先运行之前的视频抽帧和增强脚本。")
    else:
        initialize_labeling_env(TARGET_IMAGE_DIR, MY_CLASSES)