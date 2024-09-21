
import pandas as pd
import cv2
import os
import subprocess

def merge_videos_with_audio(input_video_path, annotated_video_path, output_video_path):
    # ffmpeg命令
    ffmpeg_command = [
        'ffmpeg',
        '-i', annotated_video_path,    # 输入经过标注的视频文件路径
        '-i', input_video_path,        # 输入原始视频文件路径（音频流）
        '-c:v', 'copy',                # 复制视频流
        '-c:a', 'copy',                # 复制音频流
        '-map', '0:v:0',               # 从第一个输入文件（经过标注的视频）中复制视频流
        '-map', '1:a:0',               # 从第二个输入文件（原始视频）中复制音频流
        '-shortest',                   # 保持输出视频时长与第一个输入视频相同
        output_video_path              # 输出视频文件路径
    ]

    # 执行ffmpeg命令
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"合并完成，输出文件保存为：{output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"合并视频时出现错误：{e}")

def parse_voice_match(file_path):
    """ Parses voice match data file into a DataFrame. """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            range_part, kp_id_part = line.strip().split("; ")
            start_range, end_range = map(int, range_part.split(": ")[1].split('-'))
            kp_id = kp_id_part.split(": ")[1]
            data.append((start_range, end_range, kp_id))
    return pd.DataFrame(data, columns=['start_range', 'end_range', 'kp_id'])

def parse_final_match_test(file_path):
    """ Parses final match test data file into a DataFrame. """
    data = []
    current_timestamp = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if "Timestamp:" in line:
                current_timestamp = int(line.split(": ")[1])
            elif "(" in line:
                area_part = line.split(":")[0].strip()
                kp_id_part = line.split("Knowledge_point_id: ")[1].strip()
                area = tuple(map(int, area_part.strip("()").split(", ")))
                kp_id = kp_id_part
                data.append((current_timestamp, area, kp_id))
    return pd.DataFrame(data, columns=['timestamp', 'area', 'kp_id'])

def merge_data(voice_match_df, final_match_test_df):
    """ Merges voice match data with final match test data based on overlapping time ranges and matching kp_ids. """
    # Create timestamp ranges
    timestamps = sorted(final_match_test_df['timestamp'].unique())
    timestamp_ranges = [(timestamps[i], timestamps[i + 1]) for i in range(len(timestamps) - 1)]
    timestamp_ranges.append((timestamps[-1], float('inf')))  # last range goes to infinity

    # Assign areas to timestamp ranges in a new DataFrame
    video_area_with_ranges = []
    for start, end in timestamp_ranges:
        matching_rows = final_match_test_df[final_match_test_df['timestamp'] == start]
        for _, row in matching_rows.iterrows():
            video_area_with_ranges.append((start, end - 1, row['area'], row['kp_id']))

    video_area_range_df = pd.DataFrame(video_area_with_ranges, columns=['start_range', 'end_range', 'area', 'kp_id'])

    # Merge data based on kp_id and overlapping time ranges
    output_data = []
    for _, v_row in voice_match_df.iterrows():
        matching_areas = video_area_range_df[(video_area_range_df['kp_id'] == v_row['kp_id']) &
                                             (video_area_range_df['start_range'] <= v_row['end_range']) &
                                             (video_area_range_df['end_range'] >= v_row['start_range'])]
        for _, m_row in matching_areas.iterrows():
            overlap_start = max(v_row['start_range'], m_row['start_range'])
            overlap_end = min(v_row['end_range'], m_row['end_range'])
            output_data.append(f"range: {overlap_start}-{overlap_end}; area: {m_row['area']}")

    return output_data

# 读取时间范围和区域数据
def read_areas(filename):
    areas = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(';')
            range_part = parts[0].split(':')[1].strip()
            area_part = parts[1].split(':')[1].strip().strip('()').split(',')
            start, end = map(float, range_part.split('-'))
            x1, y1, x2, y2 = map(int, area_part)
            areas.append(((start, end), (x1, y1, x2, y2)))
    return areas

# 标注视频
def annotate_video(video_path, areas, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)

        # 绘制所有当前时间内的区域
        for (start, end), (x1, y1, x2, y2) in areas:
            if start <= current_time <= end:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 绿色框

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Paths to files
video_name = '8循环'
video_folder_path = 'video_data'
voice_match_file_path = rf'video_result_data\{video_name}_data\voice_match_final.txt'
final_match_test_file_path = rf'video_result_data\{video_name}_data\final_match_merge.txt'
annotation_file_path = rf'video_result_data\{video_name}_data\attention_area.txt'
video_file_path = rf'{video_folder_path}\{video_name}.mp4'
output_video_path = rf'{video_folder_path}\{video_name}_annotated_video.mp4'
output_video_path_1 = rf'{video_folder_path}\{video_name}_annotated_video_1.mp4'

# Process the files
voice_match_df = parse_voice_match(voice_match_file_path)
final_match_test_df = parse_final_match_test(final_match_test_file_path)
merged_output = merge_data(voice_match_df, final_match_test_df)

# Write the merged output to a new file
with open(annotation_file_path, 'w', encoding='utf-8') as file:
    for line in merged_output:
        file.write(line + '\n')

# 读取区域数据
areas = read_areas(annotation_file_path)

# 标注视频
annotate_video(video_file_path, areas, output_video_path)

# 合并视频并保留带声音的最终视频
merge_videos_with_audio(video_file_path, output_video_path, output_video_path_1)

# 删除中间文件，只保留带声音的输出视频
if os.path.exists(output_video_path):
    os.remove(output_video_path)

print("Processing completed. Final video saved as:", output_video_path_1)









