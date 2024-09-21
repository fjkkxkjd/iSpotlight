import os
import re
import string
from sentence_transformers import SentenceTransformer, util
from gpt import detection_gpt
from fuzzywuzzy import fuzz
from PIL import Image

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def fuzzy_regex_match(short_sentence, long_sentence):
    short_sentence = remove_punctuation(short_sentence)
    long_sentence = remove_punctuation(long_sentence)
    pattern = re.escape(short_sentence)
    pattern = pattern.replace(r'\ ', r'(?:\s|\W)*')
    match = re.search(pattern, long_sentence, re.IGNORECASE)
    return bool(match)


def fuzz_similarity(short_text, long_text):
    score = fuzz.partial_ratio(short_text, long_text)
    return score

def parse_knowledge_content(content):
    knowledge_points = []
    kp_blocks = content.strip().split("\n\n")
    for block in kp_blocks:
        lines = block.split("\n")
        kp_dict = {}
        for line in lines:
            key, value = line.split(":", 1)
            kp_dict[key.strip()] = value.strip()
        knowledge_points.append(kp_dict)
    return knowledge_points

def read_knowledge_file(timestamp, json_folder_path):
    knowledge_file_path = os.path.join(json_folder_path, f"{timestamp}.txt")
    try:
        with open(knowledge_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return parse_knowledge_content(content)
    except Exception as e:
        return str(e)

def parse_merge_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    timestamps = re.split(r'Timestamp: (\d+)', content)[1:]
    timestamp_data = {timestamps[i]: timestamps[i+1] for i in range(0, len(timestamps), 2)}
    return timestamp_data

def expand_bbox(bbox, image_size, factor=0.2):
    x1, y1, x2, y2 = bbox
    width, height = image_size
    
    dw = (x2 - x1) * factor / 2
    dh = (y2 - y1) * factor / 2
    
    x1 = max(0, x1 - dw)
    y1 = max(0, y1 - dh)
    x2 = min(width, x2 + dw)
    y2 = min(height, y2 + dh)
    
    return (int(x1), int(y1), int(x2), int(y2))

def merge_bboxes(bboxes):
    merged = []
    while bboxes:
        base = bboxes.pop(0)
        bx1, by1, bx2, by2 = base
        to_merge = [base]
        
        for bbox in bboxes[:]:
            x1, y1, x2, y2 = bbox
            if not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2):
                to_merge.append(bbox)
                bboxes.remove(bbox)
        
        merged_bbox = (
            min(b[0] for b in to_merge),
            min(b[1] for b in to_merge),
            max(b[2] for b in to_merge),
            max(b[3] for b in to_merge)
        )
        merged.append(merged_bbox)
    
    return merged

def kp_match_data(merge_text_path, json_folder_path,original_frames_folder, object_frames_folder, output_path):
    data = parse_merge_text(merge_text_path)
    output_data = []
    
    for timestamp, contents in data.items():
        knowledge_content = read_knowledge_file(timestamp, json_folder_path)
        knowledge_txt_path = os.path.join(json_folder_path, f"{timestamp}.txt")
        if isinstance(knowledge_content, str):
            output_data.append(f"Timestamp: {timestamp}\n{contents}\n{knowledge_content}\n")
            continue
        try:
            with open(knowledge_txt_path, 'r', encoding='utf-8') as file:
                knowledge_txt = file.read()
        except FileNotFoundError:
            print(f"Knowledge file not found for timestamp {timestamp}")
            continue    

        contents_processed = contents
        detection_matches = re.findall(r'(Detection \d+): \((\d+, \d+, \d+, \d+)\)', contents)
        for match in detection_matches:
            detection_label, detection_data = match
            detection_number = detection_label.split(' ')[1].lower()
            detection_image_path = os.path.join(object_frames_folder, f"{timestamp}_detection{detection_number}.jpg")
            original_image_path = os.path.join(original_frames_folder, f"{timestamp}.jpg")
            kp_id = detection_gpt(detection_image_path, original_image_path, knowledge_txt)
            contents_processed = re.sub(rf'{detection_label}: \({detection_data}\)', rf'({detection_data}): Knowledge_point_id: {kp_id}', contents_processed)
        
        ocr_texts = re.findall(r'(OCR \d+)+: \((\d+, \d+, \d+, \d+)\) (.+)', contents)
      
        for ocr_text in ocr_texts:
            best_match = None
            best_score = -float('inf')

            identifier = ocr_text[0]
            position = ocr_text[1] 
            text = ocr_text[2]
            text_for_replace = f"{identifier}: ({position}) {text}"

            for kp in knowledge_content:
                fuzz_value = fuzz_similarity(ocr_text[2], kp['KP_description'])
                if  fuzz_value > best_score:
                    best_match = kp
                    best_score = fuzz_value

            if best_score < 80:
                for kp in knowledge_content:
                    if fuzzy_regex_match(ocr_text[2], kp['KP_description']):
                        best_match = kp
                        best_score = 100
                        
            if best_match:
                replacement_text = f"({position}): Knowledge_point_id: {best_match['Knowledge_point_id']}"
                contents_processed = contents_processed.replace(text_for_replace, replacement_text)

        image_path = os.path.join(object_frames_folder, f"{timestamp}.jpg")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            width, height = image.size
        else:
            width, height = 1920, 1080  # 默认图像尺寸

        areas = re.findall(r'\((\d+, \d+, \d+, \d+)\): Knowledge_point_id: ([\w, ]+)', contents_processed)
        area_dict = {}
        for area, kp_id in areas:
            x1, y1, x2, y2 = map(int, area.split(', '))
            expanded_bbox = expand_bbox((x1, y1, x2, y2), (width, height))
            if kp_id not in area_dict:
                area_dict[kp_id] = []
            area_dict[kp_id].append(expanded_bbox)
        
        processed_content = f"Timestamp: {timestamp}\n"
        for kp_id, bboxes in area_dict.items():
            for bbox in bboxes:
                processed_content += f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}): Knowledge_point_id: {kp_id}\n"
        
        output_data.append(processed_content)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(output_data))