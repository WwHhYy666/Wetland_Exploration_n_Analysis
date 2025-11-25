import cv2
import numpy as np
import os
import glob
import time

def guided_filter(I, p, r, eps):
    """
    导向滤波 (Guided Filter) - 用于细化透射率图，保留边缘细节
    I: 引导图像 (归一化后的原图)
    p: 输入图像 (粗略透射率图)
    r: 滤波半径
    eps: 正则化参数
    """
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q

def get_dark_channel(img, size):
    """计算暗通道图"""
    b, g, r = cv2.split(img)
    min_img = cv2.min(cv2.min(b, g), r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel

def get_atmospheric_light(img, dark_channel):
    """估算大气光值 A"""
    h, w = img.shape[:2]
    img_size = h * w
    num_pixels = int(max(img_size * 0.001, 1)) # 取前0.1%最亮的像素

    dark_vec = dark_channel.reshape(img_size)
    img_vec = img.reshape(img_size, 3)

    indices = dark_vec.argsort()[::-1]
    indices = indices[:num_pixels]

    atms_sum = np.zeros(3)
    for ind in indices:
        atms_sum += img_vec[ind]

    A = atms_sum / num_pixels
    return A

def get_transmission(img, A, size, omega=0.95):
    """估算透射率 t"""
    norm_img = img / A
    dark_channel = get_dark_channel(norm_img, size)
    transmission = 1 - omega * dark_channel
    return transmission

def recover(img, t, A, t0=0.1):
    """根据物理模型恢复图像"""
    res = np.empty(img.shape, img.dtype)
    for i in range(3):
        res[:, :, i] = (img[:, :, i] - A[i]) / np.maximum(t, t0) + A[i]
    return np.clip(res, 0, 1)

def dehaze_image(image_path, output_path):
    """
    单张图片去雾处理主函数
    """
    # 读取图片并归一化到 [0, 1]
    src = cv2.imread(image_path)
    if src is None:
        print(f"Error: 无法读取 {image_path}")
        return
    
    img = src.astype('float64') / 255.0

    # 1. 计算暗通道
    patch_size = 15
    dark = get_dark_channel(img, patch_size)

    # 2. 估算大气光 A
    A = get_atmospheric_light(img, dark)

    # 3. 估算透射率 t
    te = get_transmission(img, A, patch_size)

    # 4. 使用导向滤波细化透射率 (这是关键，否则会有方块效应)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype('float64') / 255.0
    t = guided_filter(gray, te, r=60, eps=0.0001)

    # 5. 恢复图像
    res = recover(img, t, A)

    # 保存结果
    res_uint8 = (res * 255).astype('uint8')
    cv2.imwrite(output_path, res_uint8)

def batch_process_dehaze(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 支持 jpg, png, jpeg
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_dir, "*.png"))

    total = len(image_files)
    print(f"找到 {total} 张图片，开始进行暗通道去雾处理...\n")

    start_time = time.time()
    for i, img_file in enumerate(image_files):
        filename = os.path.basename(img_file)
        save_path = os.path.join(output_dir, filename)
        
        # 打印进度
        if (i+1) % 10 == 0:
            print(f"[{i+1}/{total}] 处理中: {filename}")
        
        dehaze_image(img_file, save_path)

    end_time = time.time()
    print(f"\n处理完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"增强后的图片已保存至: {output_dir}")

# ================= 自动路径配置 =================
if __name__ == "__main__":
    # 自动推导路径：src/preprocessing -> src -> project_root
    current_script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_script_path)
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # 定义输入和输出目录
    # 输入: 上一步生成的原始图片
    INPUT_IMAGE_DIR = os.path.join(project_root, "data", "images")
    
    # 输出: 增强后的图片 (建议分开存放，方便对比)
    OUTPUT_ENHANCED_DIR = os.path.join(project_root, "data", "enhanced_images")
    
    print(f"脚本位置: {script_dir}")
    print(f"读取目录: {INPUT_IMAGE_DIR}")
    print(f"保存目录: {OUTPUT_ENHANCED_DIR}")
    print("-" * 30)
    
    if not os.path.exists(INPUT_IMAGE_DIR):
        print(f"错误: 输入目录不存在 {INPUT_IMAGE_DIR}")
    else:
        batch_process_dehaze(INPUT_IMAGE_DIR, OUTPUT_ENHANCED_DIR)