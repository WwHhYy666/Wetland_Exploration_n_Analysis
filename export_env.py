import importlib.metadata
import sys

def export_key_requirements(output_file="requirements.txt"):
    # 这里列出本项目实际用到的核心库
    # 我们之前的脚本主要依赖这些
    key_packages = [
        "ultralytics",      # YOLOv11核心
        "opencv-python",    # cv2 图像处理
        "numpy",            # 矩阵运算
        "pandas",           # 统计分析
        "matplotlib",       # 绘图
        "seaborn",          # 高级绘图
        "labelImg",         # 标注工具 (可选)
        "PyQt5",            # LabelImg依赖
        "torch",            # 深度学习框架 (虽然ultralytics会装，但显式列出更好)
        "torchvision",
        "pyyaml",           # 配置文件处理
        "tqdm"              # 进度条
    ]

    print(f"正在检测关键包版本...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for package in key_packages:
            try:
                # 获取当前环境中的版本号
                version = importlib.metadata.version(package)
                line = f"{package}=={version}"
                f.write(line + "\n")
                print(f"✅ 捕获: {line}")
            except importlib.metadata.PackageNotFoundError:
                print(f"⚠️ 警告: 当前环境未安装 {package}，已跳过")
    
    print("-" * 30)
    print(f"🎉 导出完成！文件已保存为: {output_file}")
    print("他人可以通过以下命令安装环境：")
    print(f"pip install -r {output_file}")

if __name__ == "__main__":
    export_key_requirements()