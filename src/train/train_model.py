from ultralytics import YOLO
import torch
import os

# 强制设置环境变量，减少底层库的冲突风险
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # 依然坚持使用 Small 模型
    model = YOLO('yolo11s-seg.pt') 

    print("启动 1024 分辨率稳定训练模式...")
    print("配置策略: 极低Batch + SGD + 单线程 -> 降低瞬间功耗")

    try:
        results = model.train(
            data='chenhu_seg.yaml', # 使用你做过物理扩充(复制粘贴)后的数据集
            
            # === 核心：保住 1024 的代价 ===
            imgsz=1024,      # 坚持使用 1024，确保小目标看得清
            batch=8,         # 【关键】降为 2。这是防死机的核心。
                             # 虽然速度慢，但能大幅降低 GPU/CPU 的瞬时峰值功耗。
            
            # === 优化器策略 ===
            optimizer='SGD', # 【关键】强制使用 SGD。
                             # 相比 AdamW，SGD 的计算量更小，发热更低，且显存占用更少。
            momentum=0.9,    # SGD 的标准动量
            
            # === 训练参数 ===
            epochs=150,      # 因为 Batch 小，训练波动可能稍大，多跑几轮
            patience=30,     # 早停
            
            # === 硬件保护 ===
            device=0,
            workers=2,       # 单线程，防止 CPU 过热
            amp=True,        # 混合精度 (必须开，降温神器)
            cache=False,     # 关闭 RAM 缓存
            
            # === 降低 I/O 负担 ===
            plots=True,      # 保留画图
            save_period=5,   # 每 5 轮存一次权重，减少硬盘读写频率
            
            project='Chenhu_HighRes_Stable',
            name='s_batch8_worker2_1024',
        )
    except Exception as e:
        print(f"训练发生异常: {e}")
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()