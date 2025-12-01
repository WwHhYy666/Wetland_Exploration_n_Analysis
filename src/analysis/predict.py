from ultralytics import YOLO
import cv2

def predict_wetland_plants(image_path, model_path='../models/wetland_best.pt'):
    # 加载你训练好的模型
    model = YOLO(model_path)
    
    # 预测
    results = model(image_path)
    
    # 处理结果
    for result in results:
        # 保存结果图
        result.save(filename='result.jpg') 
        # 获取分割掩码 (Masks)
        if result.masks:
            masks = result.masks.data
            # 这里可以添加代码计算植被覆盖面积
            
if __name__ == "__main__":
    predict_wetland_plants("../data/samples/test_plant.jpg")