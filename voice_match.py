import re
from collections import defaultdict
import json

def process_voice_and_final_data(voice_file_path, final_file_path, output_file_path):
    # 读取并解析 voice_match_final.txt 文件
    def load_voice_match(file_path):
        voice_data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.match(r'range: (\d+)-(\d+); kp_id: (.+)', line.strip())
                if match:
                    start, end, kp_ids = match.groups()
                    kp_ids = kp_ids.split(', ')
                    voice_data.append({
                        'start': int(start),
                        'end': int(end),
                        'kp_ids': kp_ids
                    })
        return voice_data

    # 读取并解析 final_match_merge.txt 文件
    def load_final_match(file_path):
        final_data = defaultdict(list)
        timestamps = []
        with open(file_path, 'r', encoding='utf-8') as file:
            current_timestamp = None
            for line in file:
                if line.startswith('Timestamp:'):
                    current_timestamp = int(line.strip().split(' ')[1])
                    timestamps.append(current_timestamp)
                else:
                    match = re.match(r'\((\d+), (\d+), (\d+), (\d+)\): Knowledge_point_id: (.+)', line.strip())
                    if match:
                        x1, y1, x2, y2, kp_ids = match.groups()
                        kp_ids = kp_ids.split(', ')
                        final_data[current_timestamp].append({
                            'area': (int(x1), int(y1), int(x2), int(y2)),
                            'kp_ids': kp_ids
                        })
        return final_data, timestamps

    # 根据范围合并数据
    def merge_data_by_ranges(voice_data, final_data, timestamps):
        merged_results = []

        for voice_entry in voice_data:
            start, end, voice_kp_ids = voice_entry['start'], voice_entry['end'], voice_entry['kp_ids']
            matched_areas = []

            # 查找符合时间范围的相关时间戳
            for i in range(len(timestamps) - 1):
                if start >= timestamps[i] and end <= timestamps[i + 1]:
                    relevant_timestamp = timestamps[i]

                    # 查找该时间戳下的区域是否包含匹配的知识点
                    for entry in final_data[relevant_timestamp]:
                        if any(kp in entry['kp_ids'] for kp in voice_kp_ids):
                            matched_areas.append({
                                'timestamp': relevant_timestamp,
                                'area': entry['area'],
                                'kp_ids': entry['kp_ids']
                            })
                    break
            else:
                # 处理最后一个时间戳的情况
                if start >= timestamps[-1]:
                    relevant_timestamp = timestamps[-1]
                    for entry in final_data[relevant_timestamp]:
                        if any(kp in entry['kp_ids'] for kp in voice_kp_ids):
                            matched_areas.append({
                                'timestamp': relevant_timestamp,
                                'area': entry['area'],
                                'kp_ids': entry['kp_ids']
                            })

            # 保存匹配结果
            merged_results.append({
                'time_range': f"{start}-{end}",
                'kp_ids': voice_kp_ids,
                'matched_areas': matched_areas
            })

        return merged_results

    # 加载数据
    voice_data = load_voice_match(voice_file_path)
    final_data, timestamps = load_final_match(final_file_path)

    # 合并数据
    merged_results = merge_data_by_ranges(voice_data, final_data, timestamps)

    # 保存合并结果到文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=4)

    print(f"数据已保存到 {output_file_path}")


