import cv2
import re
import os
import piexif
import math
from pathlib import Path

# ================= 配置区域 =================
# 1. 视频和SRT所在的文件夹 (输入)
VIDEO_ROOT = r"E:\Wetland_Exploration_n_Analysis\data\self_dataset\videos"

# 2. 结果输出根目录 (输出)
OUTPUT_ROOT = r"E:\Wetland_Exploration_n_Analysis\data\DOM"

# 3. 每个子任务文件夹包含的最大图片数量
# 建议 500-600 张，适合 WebODM 单次处理
IMAGES_PER_PART = 600 

# 4. 抽帧间隔 (秒)
# 1.0 表示每秒抽一帧
INTERVAL_SEC = 1.0 
# ===========================================

# 全局计数器，用于跨视频分包
GLOBAL_IMG_COUNT = 0

def normalize_exif_date(date_str):
    """
    [关键修复] 将各种格式的日期转换为 EXIF 标准格式: YYYY:MM:DD HH:MM:SS
    WebODM 严格要求使用冒号分隔
    """
    if not date_str:
        return "2024:01:01 00:00:00" # 默认值防止报错
    
    # 替换 - 和 . 为 :
    normalized = date_str.replace('-', ':').replace('.', ':')
    
    # 确保格式正确 (简单校验)
    # 如果原字符串带有毫秒 (2024:10:14 12:00:00,000)，去掉逗号后面
    if ',' in normalized:
        normalized = normalized.split(',')[0]
        
    return normalized

def decimal_to_dms(decimal):
    """将十进制经纬度转换为EXIF所需的DMS格式"""
    degrees = int(decimal)
    minutes = int((decimal - degrees) * 60)
    seconds = (decimal - degrees - minutes / 60) * 3600
    return ((degrees, 1), (minutes, 1), (int(seconds * 10000), 10000))

def parse_srt_time(time_str):
    """将SRT时间字符串转换为秒"""
    try:
        h, m, s = time_str.replace(',', '.').split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except:
        return 0.0

def parse_srt_smart(srt_path):
    """解析SRT文件，提取GPS和时间信息"""
    print(f"正在解析字幕: {os.path.basename(srt_path)}")
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        try:
            with open(srt_path, 'r', encoding='gbk', errors='ignore') as f:
                content = f.read()
        except:
            print("  -> SRT读取失败")
            return []

    blocks = re.split(r'\n\s*\n', content.strip())
    parsed_data = []
    
    # 正则表达式匹配
    # 匹配 New DJI 格式: GPS(113.12, 30.12, 10.5) ... 2024-10-14 12:00:00
    regex_new_v2 = re.compile(
        r'(\d{4}[\.-]\d{2}[\.-]\d{2} \d{2}:\d{2}:\d{2}).*?GPS\(([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\)',
        re.DOTALL | re.IGNORECASE
    )
    regex_new = re.compile(
        r'GPS\(([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\).*?(\d{4}[\.-]\d{2}[\.-]\d{2} \d{2}:\d{2}:\d{2})',
        re.DOTALL | re.IGNORECASE
    )
    # 匹配 Old DJI 格式
    regex_old = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?latitude:\s*([\d\.]+).*?longitude:\s*([\d\.]+).*?altitude:\s*([\d\.]+)', 
        re.DOTALL | re.IGNORECASE
    )

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3: continue
        
        # 提取时间轴
        time_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
        if not time_match: continue
        
        start_sec = parse_srt_time(time_match.group(1))
        end_sec = parse_srt_time(time_match.group(2))
        
        text_content = " ".join(lines[2:])
        lat, lon, alt, date_str = None, None, None, None
        
        # 尝试不同正则
        m = regex_new_v2.search(text_content)
        if m:
            date_str, lon, lat, alt = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
        
        if not lat:
            m = regex_new.search(text_content)
            if m:
                lon, lat, alt, date_str = float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)

        if not lat:
            m = regex_old.search(text_content)
            if m:
                date_str, lat, lon, alt = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
        
        if lat is not None:
            # [关键] 立即修复时间格式
            exif_time = normalize_exif_date(date_str)
            
            parsed_data.append({
                'start': start_sec,
                'end': end_sec,
                'lat': lat,
                'lon': lon,
                'alt': alt,
                'time': exif_time # 已经是 YYYY:MM:DD HH:MM:SS
            })

    print(f"  -> 解析成功: {len(parsed_data)} 条GPS记录")
    return parsed_data

def get_output_folder():
    """根据全局计数器决定当前图片应该放入哪个 part 文件夹"""
    global GLOBAL_IMG_COUNT
    # 计算当前是第几部分 (从1开始)
    part_idx = math.floor(GLOBAL_IMG_COUNT / IMAGES_PER_PART) + 1
    
    folder_name = f"wetland_proj_part{part_idx}"
    full_path = os.path.join(OUTPUT_ROOT, folder_name)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"\n>>> 创建新任务文件夹: {folder_name} (每包限额: {IMAGES_PER_PART} 张) <<<")
        
    return full_path

def find_gps_by_time(gps_data, current_time_sec):
    """查找对应时间的GPS数据"""
    for item in gps_data:
        if item['start'] <= current_time_sec < item['end']:
            return item
    
    # 容错查找 (1.5秒内)
    closest = None
    min_diff = 1.5
    for item in gps_data:
        diff = abs(item['start'] - current_time_sec)
        if diff < min_diff:
            min_diff = diff
            closest = item
    return closest

def process_single_video(video_path, srt_path):
    global GLOBAL_IMG_COUNT
    
    video_name = Path(video_path).stem
    gps_data = parse_srt_smart(srt_path)
    
    if not gps_data:
        print(f"  [跳过] 无有效GPS数据: {video_name}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * INTERVAL_SEC)
    
    print(f"  -> 视频处理中: {video_name} (FPS={fps:.1f}, 总帧数={total_frames})")

    frame_count = 0
    saved_in_video = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_count % frame_interval == 0:
            current_time = frame_count / fps
            record = find_gps_by_time(gps_data, current_time)
            
            if record:
                # 1. 获取当前应该存放的目录 (自动分包)
                current_output_dir = get_output_folder()
                
                # 2. 保存图片
                # 文件名包含绝对计数，防止重名
                filename = f"img_{GLOBAL_IMG_COUNT:05d}_{video_name}_t{current_time:.1f}.jpg"
                save_path = os.path.join(current_output_dir, filename)
                cv2.imwrite(save_path, frame)
                
                # 3. 写入 EXIF
                lat, lon, alt = record['lat'], record['lon'], record['alt']
                time_str = record['time'] # YYYY:MM:DD HH:MM:SS
                
                zeroth_ifd = {piexif.ImageIFD.Make: "DJI", piexif.ImageIFD.DateTime: time_str}
                exif_ifd = {
                    piexif.ExifIFD.DateTimeOriginal: time_str, 
                    piexif.ExifIFD.DateTimeDigitized: time_str
                }
                gps_ifd = {
                    piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
                    piexif.GPSIFD.GPSLatitude: decimal_to_dms(abs(lat)),
                    piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
                    piexif.GPSIFD.GPSLongitude: decimal_to_dms(abs(lon)),
                    piexif.GPSIFD.GPSAltitudeRef: 0, # 0 = Sea level
                    piexif.GPSIFD.GPSAltitude: (int(alt * 100), 100)
                }
                
                try:
                    exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd}
                    exif_bytes = piexif.dump(exif_dict)
                    piexif.insert(exif_bytes, save_path)
                except Exception as e:
                    print(f"EXIF写入错误: {e}")

                saved_in_video += 1
                GLOBAL_IMG_COUNT += 1
            
        frame_count += 1

    cap.release()
    print(f"  -> {video_name} 完成: 贡献了 {saved_in_video} 张图片")

def main():
    if not os.path.exists(VIDEO_ROOT):
        print(f"错误: 找不到视频文件夹 {VIDEO_ROOT}")
        return

    # 获取所有mp4文件
    video_files = [f for f in os.listdir(VIDEO_ROOT) if f.lower().endswith(('.mp4', '.mov'))]
    video_files.sort() # 排序，确保处理顺序
    
    print(f"=== 开始处理湿地数据 ===")
    print(f"源目录: {VIDEO_ROOT}")
    print(f"目标目录: {OUTPUT_ROOT}")
    print(f"分包策略: 每 {IMAGES_PER_PART} 张图片创建一个新文件夹\n")

    for v_file in video_files:
        video_path = os.path.join(VIDEO_ROOT, v_file)
        
        # 寻找同名SRT
        srt_name = os.path.splitext(v_file)[0] + ".srt"
        srt_path = os.path.join(VIDEO_ROOT, srt_name)
        
        if os.path.exists(srt_path):
            process_single_video(video_path, srt_path)
        else:
            print(f"[警告] 视频 {v_file} 缺少对应的 SRT 文件，跳过处理。")

    print(f"\n=== 全部完成 ===")
    print(f"总计生成图片: {GLOBAL_IMG_COUNT}")
    print(f"查看输出目录: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()