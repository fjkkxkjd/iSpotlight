import os
import pandas as pd
import json
import cv2
from gpt import voice_gpt

def get_video_length(video_path):
    """ Get the length of the video in seconds. """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return 0
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return length

def update_ranges_in_json_files(json_files_dir, timestamps_file_path, video_length):
    with open(timestamps_file_path, 'r', encoding='utf-8') as file:
        timestamps = [int(line.strip()) for line in file.readlines()]

    for json_filename in os.listdir(json_files_dir):
        if json_filename.endswith('.json'):
            json_file_path = os.path.join(json_files_dir, json_filename)
            assigned_timestamp = int(json_filename[:-5])

            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if not data:
                continue

            first_sentence = data[0]
            _, end_range = map(int, first_sentence["range"].split('-'))
            first_sentence["range"] = f"{assigned_timestamp}-{end_range}"
            
            if assigned_timestamp >= end_range:
                first_sentence["range"] = f"{assigned_timestamp}-{assigned_timestamp}"
            
            next_timestamp = None
            for ts in timestamps:
                if ts > assigned_timestamp:
                    next_timestamp = ts
                    break

            if next_timestamp:
                last_sentence = data[-1]
                start_range, _ = map(int, last_sentence["range"].split('-'))
                last_sentence["range"] = f"{start_range}-{next_timestamp}"
            else:
                last_sentence = data[-1]
                start_range, _ = map(int, last_sentence["range"].split('-'))
                last_sentence["range"] = f"{start_range}-{int(video_length*1000)}"

            with open(json_file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

def process_result_json_folder(input_folder, output_file):
    # Get a list of all JSON files in the folder
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    # Sort the files by numeric order assuming the filenames are numbers ending with .json
    json_files.sort(key=lambda f: int(f.split('.')[0]))

    with open(output_file, 'w', encoding='utf-8') as output:
        for json_file in json_files:
            file_path = os.path.join(input_folder, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    messages = data["choices"][0]["message"]["content"]
                    # Assuming the content is wrapped in code block markers, strip them off
                    clean_content = messages.strip("```json\n").strip("\n```")
                    # Convert the cleaned content back to JSON
                    content_data = json.loads(clean_content)
                    # Extract detailed information and write to file
                    for item in content_data:
                        range_info = item["range"]
                        kp_id = item.get("kp_id", "No KP ID")
                        output.write(f"range: {range_info}; kp_id: {kp_id}\n")
            except Exception as e:
                output.write(f"Error processing file {file_path}: {str(e)}\n")

def process_data_and_prepare_for_voice_gpt(tsv_file_path, timestamps_file_path, json_files_dir, result_json_files_dir, output_function, kp_json_path, video_file_path, output_frames_folder):
    # 获取视频长度
    video_length = get_video_length(video_file_path)
    print(f"Video length: {video_length} seconds")
    
    # 创建输出目录
    if not os.path.exists(json_files_dir):
        os.makedirs(json_files_dir)
    
    if not os.path.exists(result_json_files_dir):
        os.makedirs(result_json_files_dir)
    
    # 读取 TSV 数据到 DataFrame
    tsv_data = pd.read_csv(tsv_file_path, sep='\t')
    
    # 读取时间戳文件到列表
    with open(timestamps_file_path, 'r', encoding='utf-8') as file:
        timestamps = [int(line.strip()) for line in file.readlines()]
    # 存储提取的文本段的字典
    segments = {}

    # 遍历 TSV 数据行并根据时间戳范围分配到合适的 JSON 文件中
    for index, row in tsv_data.iterrows():
        start_range = row['start']
        end_range = row['end']
        text = row['text']
        # 找到合适的时间戳
        assigned_timestamp = None
        for i in range(len(timestamps)):
            if end_range <= timestamps[i]:
                if i == 0:
                    # 如果 end_range 小于等于第一个时间戳，分配给第一个时间戳
                    assigned_timestamp = timestamps[i]
                elif start_range >= timestamps[i-1]:
                    assigned_timestamp = timestamps[i-1]
                elif start_range < timestamps[i-1]:
                    if (end_range - timestamps[i - 1]) > (timestamps[i - 1] - start_range):
                        assigned_timestamp = timestamps[i-1]
                        start_range = assigned_timestamp
                    elif (end_range - timestamps[i - 1]) <= (timestamps[i - 1] - start_range):
                        assigned_timestamp = timestamps[i-2]
                        end_range = timestamps[i-1]
                break

        if assigned_timestamp is None:
            # 如果没有合适的时间戳，使用视频的最后一刻
            # assigned_timestamp = int(video_length*1000)
            assigned_timestamp = timestamps[len(timestamps)-1]

        # 使用分配的时间戳作为 JSON 文件名
        json_filename = f"{assigned_timestamp}.json"
        json_path = os.path.join(json_files_dir, json_filename)

        # 将文本写入 JSON 文件，包括范围信息
        if assigned_timestamp not in segments:
            segments[assigned_timestamp] = []

        range_info = f"{start_range}-{end_range}"
        segments[assigned_timestamp].append({"text": text, "range": range_info})

        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(segments[assigned_timestamp], json_file, ensure_ascii=False, indent=4)

    # 补全没有分配到文本段的时间戳
    for i in range(len(timestamps)):
        if timestamps[i] not in segments:
            closest_text = None
            min_diff = float('inf')
            
            # 找到合适的时间戳
            next_timestamp = None
            for ts in timestamps:
                if ts > timestamps[i]:
                    next_timestamp = ts
                    break
            
            json_content = []
            
            for index, row in tsv_data.iterrows():
                start_range = row['start']
                end_range = row['end']
                text = row['text']
                if start_range < timestamps[i] and (closest_text is None or abs(start_range - timestamps[i]) < min_diff):
                    closest_text = text
                    min_diff = abs(start_range - timestamps[i])    

                start_range = timestamps[i]

                if next_timestamp:
                    end_range = next_timestamp
                else:
                    end_range = int(video_length)

            range_info = f"{start_range}-{end_range}"    
            json_content.append({
                    "text": closest_text,
                    "range": range_info
                })
            # 创建对应的 JSON 文件
            json_filename = f"{timestamps[i]}.json"
            json_path = os.path.join(json_files_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_content, json_file, ensure_ascii=False, indent=4)
    
    update_ranges_in_json_files(json_files_dir, timestamps_file_path, video_length)            
    
    # 处理生成的 JSON 文件，调用 voice_gpt 处理并保存响应
    for filename in os.listdir(json_files_dir):
        if filename.endswith('.json'):
            json_file_path = os.path.join(json_files_dir, filename)
            txt_file_path = os.path.join(kp_json_path, f"{filename[:-5]}.txt")  # Assuming txt files are in 'txt_files' directory
            ppt_image_path = os.path.join(output_frames_folder, f"{filename[:-5]}.jpg")
            if os.path.exists(txt_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as json_file, open(txt_file_path, 'r', encoding='utf-8') as txt_file:
                    json_data = json.load(json_file)
                    text_content = ' '.join(f'sentence:"{text["text"]}", range:"{text["range"]}"' for text in json_data)
                    txt_content = txt_file.read().strip()
                    # Call voice_gpt function to process content
                    response = output_function(txt_content, text_content, ppt_image_path)
                    
                    # Save response to new JSON file in result_json_files_dir
                    result_json_path = os.path.join(result_json_files_dir, filename)
                    with open(result_json_path, 'w', encoding='utf-8') as result_json_file:
                        json.dump(response, result_json_file, ensure_ascii=False, indent=4)
                    print(f"Response saved to {result_json_path}")
            else:
                print(f"TXT file not found for {filename}")
