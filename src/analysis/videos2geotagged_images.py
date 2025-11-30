import cv2
import re
import os
import piexif
from pathlib import Path

# ================= 配置区域 =================
VIDEOS_DIR = 'videos'          # 视频文件夹
TAGS_DIR = 'tags'              # SRT文件夹
OUTPUT_DIR = 'dataset_images'  # 结果保存路径
INTERVAL_SEC = 1               # 抽帧间隔（建议设为1，因为新SRT每秒才更新一次）
# ===========================================

def decimal_to_dms(decimal):
    """将十进制经纬度转换为EXIF所需的DMS格式"""
    degrees = int(decimal)
    minutes = int((decimal - degrees) * 60)
    seconds = (decimal - degrees - minutes / 60) * 3600
    return ((degrees, 1), (minutes, 1), (int(seconds * 10000), 10000))

def parse_srt_time(time_str):
    """将SRT时间字符串 (00:00:01,000) 转换为秒 (float)"""
    # 格式: HH:MM:SS,mmm
    h, m, s = time_str.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def parse_srt_smart(srt_path):
    """
    智能解析SRT，支持多种DJI格式
    返回列表: [{'start': 1.0, 'end': 2.0, 'lat': 30.x, 'lon': 113.x, 'alt': 15.0, 'time': 'str'}, ...]
    """
    print(f"正在解析字幕文件: {srt_path}")
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        with open(srt_path, 'r', encoding='gbk', errors='ignore') as f:
            content = f.read()

    # 1. 分割SRT块 (利用空行分割)
    # 兼容不同换行符
    blocks = re.split(r'\n\s*\n', content.strip())
    
    parsed_data = []
    
    # 预编译两种正则
    # 格式A (Old): [latitude: 30.3] ...
    regex_old = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?latitude:\s*([\d\.]+).*?longitude:\s*([\d\.]+).*?altitude:\s*([\d\.]+)', 
        re.DOTALL | re.IGNORECASE
    )
    # 格式B (New): GPS(113.8, 30.3, 15) ...
    # 注意：新格式中 GPS顺序通常是 (经度, 纬度, 高度) -> (Lon, Lat, Alt)
    regex_new = re.compile(
        r'GPS\(([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\).*?(\d{4}[\.-]\d{2}[\.-]\d{2} \d{2}:\d{2}:\d{2})',
        re.DOTALL | re.IGNORECASE
    )
    # 有些时候日期在GPS前面，做一个变体
    regex_new_v2 = re.compile(
        r'(\d{4}[\.-]\d{2}[\.-]\d{2} \d{2}:\d{2}:\d{2}).*?GPS\(([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\)',
        re.DOTALL | re.IGNORECASE
    )

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3: continue
        
        # 提取时间轴 00:00:00,000 --> 00:00:01,000
        time_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
        if not time_match: continue
        
        start_sec = parse_srt_time(time_match.group(1))
        end_sec = parse_srt_time(time_match.group(2))
        
        text_content = " ".join(lines[2:]) # 合并剩余行方便正则
        
        # 尝试匹配格式
        lat, lon, alt, date_str = None, None, None, None
        
        # 尝试 New V2 (日期在前)
        m = regex_new_v2.search(text_content)
        if m:
            date_str = m.group(1).replace('.', '-') # 统一格式
            lon = float(m.group(2))
            lat = float(m.group(3))
            alt = float(m.group(4))
        
        # 尝试 New (GPS在前，日期在后)
        if not lat:
            m = regex_new.search(text_content)
            if m:
                lon = float(m.group(1))
                lat = float(m.group(2))
                alt = float(m.group(3))
                date_str = m.group(4).replace('.', '-')

        # 尝试 Old
        if not lat:
            m = regex_old.search(text_content)
            if m:
                date_str = m.group(1)
                lat = float(m.group(2))
                lon = float(m.group(3))
                alt = float(m.group(4))
        
        if lat is not None:
            parsed_data.append({
                'start': start_sec,
                'end': end_sec,
                'lat': lat,
                'lon': lon,
                'alt': alt,
                'time': date_str
            })

    print(f"  -> 解析成功: 获取 {len(parsed_data)} 条GPS记录")
    return parsed_data

def find_gps_by_time(gps_data, current_time_sec):
    """根据视频当前秒数，查找匹配的SRT记录"""
    # 简单遍历或二分查找。由于数据量不大，遍历即可。
    # 优先找 start <= t < end
    for item in gps_data:
        if item['start'] <= current_time_sec < item['end']:
            return item
    
    # 如果找不到(比如刚开始0.5s，但字幕从1.0s开始)，找最近的一个
    # 这里的阈值设为 1.5秒，如果差距太大则认为无效
    closest = None
    min_diff = 1000
    
    for item in gps_data:
        diff = abs(item['start'] - current_time_sec)
        if diff < min_diff:
            min_diff = diff
            closest = item
            
    if min_diff < 1.5:
        return closest
    return None

def process_single_video(video_path, srt_path, output_root):
    video_name = Path(video_path).stem
    gps_data = parse_srt_smart(srt_path)
    
    if not gps_data:
        print(f"  [跳过] {video_name}: 无法从SRT提取有效GPS")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * INTERVAL_SEC) # 每隔多少帧取一张
    
    print(f"  -> 处理中: {video_name} (FPS={fps:.1f})")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_count % frame_interval == 0:
            # 计算当前视频时间的秒数
            current_time = frame_count / fps
            
            # 按时间去查GPS
            record = find_gps_by_time(gps_data, current_time)
            
            if record:
                filename = f"{video_name}_t{current_time:.1f}_{frame_count}.jpg"
                save_path = os.path.join(output_root, filename)
                cv2.imwrite(save_path, frame)
                
                # 写入EXIF
                lat, lon, alt = record['lat'], record['lon'], record['alt']
                time_str = record['time']
                
                zeroth_ifd = {piexif.ImageIFD.Make: "DJI", piexif.ImageIFD.DateTime: time_str}
                exif_ifd = {piexif.ExifIFD.DateTimeOriginal: time_str, piexif.ExifIFD.DateTimeDigitized: time_str}
                gps_ifd = {
                    piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
                    piexif.GPSIFD.GPSLatitude: decimal_to_dms(abs(lat)),
                    piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
                    piexif.GPSIFD.GPSLongitude: decimal_to_dms(abs(lon)),
                    piexif.GPSIFD.GPSAltitude: (int(alt * 100), 100)
                }
                piexif.insert(piexif.dump({"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd}), save_path)
                saved_count += 1
            else:
                # 只有当时间偏差太大找不到GPS时才警告（偶尔几帧没对上可以忽略）
                pass
            
        frame_count += 1

    cap.release()
    print(f"  -> {video_name} 完成: 生成 {saved_count} 张图")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    video_files = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    print(f"检测到 {len(video_files)} 个视频，开始处理...\n")

    for v_file in video_files:
        video_path = os.path.join(VIDEOS_DIR, v_file)
        srt_name = os.path.splitext(v_file)[0] + ".srt"
        srt_path = os.path.join(TAGS_DIR, srt_name)
        
        # 简单容错查找SRT
        if not os.path.exists(srt_path):
             candidates = [f for f in os.listdir(TAGS_DIR) if f.lower() == srt_name.lower()]
             if candidates: srt_path = os.path.join(TAGS_DIR, candidates[0])
        
        if os.path.exists(srt_path):
            process_single_video(video_path, srt_path, OUTPUT_DIR)
        else:
            print(f"[警告] 缺失字幕: {v_file}")

if __name__ == "__main__":
    main()