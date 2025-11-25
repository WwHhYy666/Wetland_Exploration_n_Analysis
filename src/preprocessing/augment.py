import cv2
import numpy as np
import os
import glob
import random

def adjust_brightness(image, factor):
    """
    调整亮度
    factor > 1 为变亮，factor < 1 为变暗
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 转换类型以防止溢出
    v = v.astype('float64')
    v = v * factor
    v = np.clip(v, 0, 255).astype('uint8')
    
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def augment_image(image_path, output_dir):
    """
    对单张图片进行增强并保存
    """
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    img = cv2.imread(image_path)
    if img is None:
        return

    # 1. 保存原图 (必须保留)
    cv2.imwrite(os.path.join(output_dir, f"{name}_orig{ext}"), img)

    # ================= 策略配置 =================
    # 我们并不希望每张图都生成所有变体，那样数据量太大了。
    # 这里采用“概率触发”机制。
    
    # 2. 水平翻转 (Horizontal Flip)
    # 对于湿地植被，左右翻转是完全合理的物理场景
    if random.random() < 0.5:
        flip_h = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(output_dir, f"{name}_flipH{ext}"), flip_h)

    # 3. 垂直翻转 (Vertical Flip)
    # 仅适用于垂直俯拍 (Nadir view)。如果是倾斜拍摄，请注释掉这段代码。
    if random.random() < 0.5:
        flip_v = cv2.flip(img, 0)
        cv2.imwrite(os.path.join(output_dir, f"{name}_flipV{ext}"), flip_v)

    # 4. 旋转 90度/180度/270度
    # 这种旋转不会丢失图像信息，也不会产生黑边
    # 随机选择一种旋转方式，或者不旋转
    rotate_code = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
    if rotate_code is not None:
        rotated = cv2.rotate(img, rotate_code)
        suffix_map = {
            cv2.ROTATE_90_CLOCKWISE: "rot90",
            cv2.ROTATE_180: "rot180",
            cv2.ROTATE_90_COUNTERCLOCKWISE: "rot270"
        }
        cv2.imwrite(os.path.join(output_dir, f"{name}_{suffix_map[rotate_code]}{ext}"), rotated)

    # 5. 亮度变化 (Brightness)
    # 模拟云层遮挡(变暗)或强光(变亮)
    # 随机在 [0.8, 1.2] 之间波动
    if random.random() < 0.3: # 30%的概率触发
        brightness_factor = random.uniform(0.7, 1.3)
        bright_img = adjust_brightness(img, brightness_factor)
        cv2.imwrite(os.path.join(output_dir, f"{name}_bright{brightness_factor:.2f}{ext}"), bright_img)

def batch_augment(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 支持常见格式
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_dir, "*.png"))
    
    total = len(image_files)
    print(f"源数据集共 {total} 张图片，开始增强...\n")

    for i, img_path in enumerate(image_files):
        augment_image(img_path, output_dir)
        
        if (i+1) % 50 == 0:
            print(f"已处理: {i+1}/{total} (生成数量会多于此数)")

    # 统计最终数量
    final_count = len(glob.glob(os.path.join(output_dir, "*.*")))
    print(f"\n增强完成！")
    print(f"源文件数: {total}")
    print(f"增强后文件总数: {final_count}")
    print(f"保存位置: {output_dir}")

# ================= 路径配置 =================
if __name__ == "__main__":
    current_script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_script_path)
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # 输入: 上一步去雾后的图片
    INPUT_DIR = os.path.join(project_root, "data", "enhanced_images")
    
    # 输出: 最终用于标注的数据集 (Final Dataset)
    OUTPUT_DIR = os.path.join(project_root, "data", "final_dataset_images")
    
    print(f"脚本位置: {script_dir}")
    print(f"读取目录: {INPUT_DIR}")
    print(f"保存目录: {OUTPUT_DIR}")
    print("-" * 30)
    
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 找不到输入目录 {INPUT_DIR}")
    else:
        batch_augment(INPUT_DIR, OUTPUT_DIR)