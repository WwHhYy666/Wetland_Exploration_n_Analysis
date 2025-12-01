# 🌿 Wetland Exploration & Analysis (沉湖湿地植被考察与分析)

<!-- Badges 徽章区 -->
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv11](https://img.shields.io/badge/Model-YOLOv11s--seg-green)
![Task](https://img.shields.io/badge/Task-Instance%20Segmentation-orange)
![Framework](https://img.shields.io/badge/Framework-Ultralytics-blueviolet)

> **基于 YOLOv11s-seg 的沉湖湿地 6 种典型植被自动化识别与实例分割项目。**

---

## 📖 项目背景 | Introduction

沉湖湿地作为国际重要湿地，其植被分布的动态变化对生态监测至关重要。本项目旨在通过计算机视觉技术，解决传统人工考察效率低、量化难的问题。

我们采集了大量无人机航拍影像，训练了一个 **YOLOv11s-seg** 实例分割模型，能够对湿地内的 **6 种优势植物**（如芦苇、苔草、菰等）进行像素级分割，为后续的生物量统计和生态分析提供数据支撑。

---

## 💾 数据集 | Dataset

由于原始航拍视频和高分辨率图像数据量巨大（超过 GitHub 限制），本仓库仅保留少量示例数据。

### 1. 数据获取
完整的数据集（包含 YOLO 格式的 txt 标签与原始图像）已托管至网盘：

- **📥 下载地址**: [点击跳转百度网盘](#) (请替换此处链接)
- **🔑 提取码**: `xxxx` (请替换此处提取码)

### 2. 数据准备
下载并解压后，请将数据放置在根目录的 `data/` 文件夹下，保持以下目录结构，以便代码自动读取：

```text
Wetland_Exploration_n_Analysis/
└── data/
    ├── samples/             # Git仓库中自带的少量测试图
    ├── images/              # [下载] 训练与验证图片
    │   ├── train/
    │   └── val/
    └── labels/              # [下载] 对应的分割标签
        ├── train/
        └── val/
```

## 📂 目录结构 | Directory Structure

本项目采用“配置-数据-代码”分离的工程化结构：

```text
├── configs/                 # ⚙️ 配置文件 (data.yaml, 训练超参数)
├── data/                    # 💾 数据存放区 (已忽略大文件)
├── docs/                    # 📄 项目文档与技术报告
├── models/                  # 🧠 模型权重 (如 wetland_best.pt)
├── notebooks/               # 📓 探索性分析 (Jupyter Notebooks)
├── runs/                    # 📈 训练日志、TensorBoard 数据
├── src/                     # 🛠️ 源代码
│   ├── train.py             # 训练入口
│   ├── predict.py           # 推理入口
│   ├── val.py               # 评估入口
│   └── utils.py             # 工具类
├── requirements.txt         # 📦 依赖包列表
└── README.md                # 📘 项目主页
```

## 🛠️ 环境安装 | Installation

推荐使用 Conda 创建独立的虚拟环境：

```code

# 1. 创建环境 (Python 3.9)
conda create -n wetland python=3.9 -y

# 2. 激活环境
conda activate wetland

# 3. 安装依赖 (包含 PyTorch 和 Ultralytics)
pip install -r requirements.

```

## 🚀 使用指南 | Usage
1. 推理预测 (Inference)
使用训练好的模型对新的航拍照片或视频进行分割：

```code

# 图片预测 (结果将保存在 runs/segment/predict/)
python src/predict.py --source data/samples/test.jpg --weights models/wetland_best.pt

# 视频预测
python src/predict.py --source data/raw/survey_video.mp4 --weights models/wetland_best.pt --save

```

2. 模型训练 (Training)
如果您需要基于新数据重新训练模型：

```code

# 单卡训练 (确保 configs/data.yaml 配置正确)
python src/train.py --config configs/data.yaml --model yolov11s-seg.pt --epochs 200

```

3. 模型验证 (Validation)
评估模型在验证集上的精度（mAP）：

```code

python src/val.py --weights models/wetland_best.pt --data configs/data.yaml

```

## 📊 实验结果 | Results
1. 可视化效果
 (请在此处插入一张您的模型推理效果图，例如：)
![alt text](data/samples/demo_result.jpg)
2. 性能指标
类别 (Class)	Precision (P)	Recall (R)	mAP@50	mAP@50-95
All	0.XX	0.XX	0.XX	0.XX
芦苇 (Reed)	0.XX	0.XX	0.XX	0.XX
苔草 (Sedge)	0.XX	0.XX	0.XX	0.XX
菰 (Zizania)	0.XX	0.XX	0.XX	0.XX