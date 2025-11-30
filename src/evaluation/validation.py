from ultralytics import YOLO

model = YOLO('best.pt')

# 使用 generator (生成器) 模式
# 注意：stream=True 时，必须用 for 循环遍历结果，否则推理不会开始
results = model.predict(
    source='test_videos/',  # 你的图片文件夹
    save=True,      # 保存结果
    imgsz=1024,     # 保持高清
    stream=True,    # 【关键】开启流式模式，用完即扔，不占内存
    device=0,       # 使用 GPU 推理
    conf=0.25
)

# 必须遍历，流程才会执行
for result in results:
    pass  # 可以在这里加一行 print("Processed one image") 来监控进度