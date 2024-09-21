import re
import os
import json

# 假设图像宽度和高度的已知值
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# 提取时间戳
def extract_timestamp_from_filename(filename):
    timestamp_match = re.search(r'EyeTracking_(\d+)\.txt', filename)
    if timestamp_match:
        return timestamp_match.group(1)
    else:
        return "unknown"

# 加载注视数据并自动提取clipname
def load_gaze_data(file_path):
    gaze_data = {}
    frame_data = {}
    clipname = None
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # 解析帧信息
        position_match = re.search(r'rawImageWorldPosition: \(([\d.]+), ([\d.]+), ([\d.]+)\)', content)
        width_match = re.search(r'rawImageworldWidth: ([\d.]+)', content)
        height_match = re.search(r'rawImageworldHeight: ([\d.]+)', content)
        if position_match:
            frame_data['position'] = (float(position_match.group(1)), float(position_match.group(2)), float(position_match.group(3)))
        if width_match:
            frame_data['width'] = float(width_match.group(1))
        if height_match:
            frame_data['height'] = float(height_match.group(1))
        frame_data['screen_width'] = IMAGE_WIDTH 
        frame_data['screen_height'] = IMAGE_HEIGHT
        
        # 解析注视数据并提取clipname
        gaze_matches = re.finditer(r'CurrentTime: ([\d.]+);Clipname: ([\w_-]+)\nposition: \(([\d.]+), ([\d.]+), ([\d.]+)\)', content)
        for match in gaze_matches:
            time = float(match.group(1))
            clipname = match.group(2)  # 自动获取clipname
            position = (float(match.group(3)), float(match.group(4)))
            if clipname not in gaze_data:
                gaze_data[clipname] = {}
            gaze_data[clipname].setdefault(time, []).append(position)
        
        print(f'Frame Data: {frame_data}')
        print(f'Gaze Data: {gaze_data}')
    return gaze_data, frame_data, clipname

# 生成 gaze_log 文件
def process_gaze_log(output_folder, gaze_data, frame_data, timestamp, clipname):
    if not gaze_data:
        raise ValueError("No gaze data was provided to process.")

    log_path = None
    for clipname, points in gaze_data.items():
        log_path = os.path.join(output_folder, f"gaze_log_{clipname}_{timestamp}.txt")
        log_lines = []
        for time, positions in points.items():
            for point in positions:
                x_pixel = int((point[0] - frame_data['position'][0] + frame_data['width'] / 2) * frame_data['screen_width'] / frame_data['width'])
                y_pixel = int((frame_data['height'] / 2 - point[1] + frame_data['position'][1]) * frame_data['screen_height'] / frame_data['height'])
                log_lines.append(f"Time: {time:.2f}s, Pixel Position: ({x_pixel}, {y_pixel})\n")
        try:
            with open(log_path, 'w') as log_file:
                log_file.writelines(log_lines)
            print(f"Gaze log processing complete for: {clipname}")
            print(log_path)
        except IOError as e:
            print(f"Failed to write to {log_path}: {e}")

    if log_path is None:
        raise ValueError("No gaze data was processed, ensure input data is correct.")

    return log_path

# 处理眼动追踪文件，生成 gaze_log 并自动获取 merged_results_path
def process_eye_tracking_file(input_file):
    output_folder = os.path.dirname(input_file)
    timestamp = extract_timestamp_from_filename(os.path.basename(input_file))
    gaze_data, frame_data, clipname = load_gaze_data(input_file)
    if clipname is None:
        raise ValueError("No clipname found in the file. Ensure the input file is correctly formatted.")
    gaze_log_path = process_gaze_log(output_folder, gaze_data, frame_data, timestamp, clipname)
    print(f"Complete processing for file: {input_file}")

    # 根据 clipname 自动构建 merged_results_path
    merged_results_path = rf'D:\yolov9\video_result_data\{clipname}_data_test\merged_results.json'
    if not os.path.exists(merged_results_path):
        raise FileNotFoundError(f"merged_results.json not found at {merged_results_path}")

    return gaze_log_path, merged_results_path

# 读取 gaze_log 文件
def parse_gaze_log(file_path):
    gaze_data = {}
    time_pattern = re.compile(r'Time: ([\d\.]+)s')
    position_pattern = re.compile(r'Pixel Position: \((\d+), (\d+)\)')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            time_match = time_pattern.search(line)
            pos_match = position_pattern.search(line)
            
            if time_match and pos_match:
                time_in_seconds = int(float(time_match.group(1)))
                x = int(pos_match.group(1))
                y = int(pos_match.group(2))
                y = IMAGE_HEIGHT - y
                
                if time_in_seconds not in gaze_data:
                    gaze_data[time_in_seconds] = []
                gaze_data[time_in_seconds].append((x, y))
    
    return gaze_data

# 读取 merged_results.json 文件，并扩展区域
def parse_merged_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        merged_results = json.load(file)

    for entry in merged_results:
        for area in entry['matched_areas']:
            x1, y1, x2, y2 = area['area']
            x1 = max(0, x1 - 80)
            y1 = max(0, y1 - 80)
            x2 = min(IMAGE_WIDTH - 1, x2 + 80)
            y2 = min(IMAGE_HEIGHT - 1, y2 + 80)
            area['area'] = [x1, y1, x2, y2]

    return merged_results

# 判断点是否在区域内
def is_point_in_area(point, area):
    x, y = point
    return area[0] <= x <= area[2] and area[1] <= y <= area[3]

# 根据注视点和区域判断模式
def get_mode_for_second(gaze_data, merged_results, second, prev_kp_ids):
    if second not in gaze_data:
        return 'other', prev_kp_ids
    
    gaze_points = gaze_data[second]
    total_points = len(gaze_points)

    current_kp_ids = None
    next_kp_ids = None
    in_current_area_count = 0
    in_prev_area_count = 0
    in_next_area_count = 0
    is_boundary_time = False

    for i, entry in enumerate(merged_results):
        time_range = entry['time_range'].split('-')
        start_time, end_time = int(time_range[0]) // 1000, int(time_range[1]) // 1000
        if start_time <= second < end_time:
            current_kp_ids = entry['kp_ids']
            matched_area = entry['matched_areas']
            in_current_area_count = sum(
                1 for point in gaze_points if any(is_point_in_area(point, area['area']) for area in matched_area)
            )
        
        if (start_time == second or end_time == second) and i < len(merged_results) - 1:
            next_entry = merged_results[i + 1]
            next_kp_ids = next_entry['kp_ids']
            if current_kp_ids != next_kp_ids:
                is_boundary_time = True

        if i < len(merged_results) - 1:
            next_entry = merged_results[i + 1]
            next_time_range = next_entry['time_range'].split('-')
            next_start_time, next_end_time = int(next_time_range[0]) // 1000, int(next_time_range[1]) // 1000

            if next_start_time == end_time:
                next_kp_ids = next_entry['kp_ids']
                in_next_area_count = sum(
                    1 for point in gaze_points if any(is_point_in_area(point, area['area']) for area in next_entry['matched_areas'])
                )
        
        if prev_kp_ids and i > 0:
            prev_entry = merged_results[i - 1]
            in_prev_area_count = sum(
                1 for point in gaze_points if any(is_point_in_area(point, area['area']) for area in prev_entry['matched_areas'] if set(area['kp_ids']).intersection(prev_kp_ids))
            )

    if in_current_area_count > total_points / 2 and in_current_area_count > 35:
        return 'follow', current_kp_ids
    
    if is_boundary_time and (in_current_area_count + in_prev_area_count) > 35 and (in_current_area_count + in_prev_area_count) > total_points / 2:
        return 'follow', current_kp_ids

    if in_prev_area_count > total_points / 2 and in_prev_area_count > 35:
        return 'backtrack', current_kp_ids
    
    if in_next_area_count > 35 and in_next_area_count > total_points / 2:
        return 'forward', next_kp_ids
    
    return 'other', prev_kp_ids

# 处理 gaze_log 数据
def process_gaze_data(gaze_log_path, merged_results_path, output_path):
    gaze_data = parse_gaze_log(gaze_log_path)
    merged_results = parse_merged_results(merged_results_path)
    
    results = []
    mode_counts = {'follow': 0, 'backtrack': 0, 'other': 0, 'forward': 0}
    prev_kp_ids = None

    total_seconds = max(gaze_data.keys()) - min(gaze_data.keys()) + 1
    
    for second in range(min(gaze_data.keys()), max(gaze_data.keys()) + 1):
        mode, prev_kp_ids = get_mode_for_second(gaze_data, merged_results, second, prev_kp_ids)
        results.append(f"{second}: {mode}")
        mode_counts[mode] += 1
    
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(results))
    
    with open(output_path, 'a', encoding='utf-8') as file:
        file.write("\nMode Proportions:\n")
        for mode, count in mode_counts.items():
            proportion = (count / total_seconds) * 100
            file.write(f"{mode}: {count} ({proportion:.2f}%)\n")
            print(f"{mode}: {count} ({proportion:.2f}%)")

# 遍历用户文件夹，生成 gaze_log 并处理
def process_all_users_with_gazelog(folder_a_path):
    for root, dirs, files in os.walk(folder_a_path):
        for dir_name in dirs:
            ch_folder_path = os.path.join(root, dir_name)
            if os.path.exists(ch_folder_path):
                for file_name in os.listdir(ch_folder_path):
                    if file_name.startswith('EyeTracking') and file_name.endswith('.txt'):
                        input_file_path = os.path.join(ch_folder_path, file_name)

                        # 生成 gaze_log 并获取对应的 merged_results_path
                        gaze_log_path, merged_results_path = process_eye_tracking_file(input_file_path)
                        print(f"Generated gaze_log: {gaze_log_path}")

                        # 使用生成的 gaze_log 文件进行后续处理
                        if gaze_log_path:
                            output_path = os.path.join(root, dir_name, 'gaze_mode.txt')
                            process_gaze_data(gaze_log_path, merged_results_path, output_path)
                            print(f"Processed {gaze_log_path}, output saved to {output_path}")

# 调用函数处理所有用户文件夹
folder_a_path = r"user_study_eyetracking_attention"
process_all_users_with_gazelog(folder_a_path)
