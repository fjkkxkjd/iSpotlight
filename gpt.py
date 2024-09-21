import os
import base64
import requests
import json
import time 
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import re
# OpenAI API Key
api_key = 'your_openai_api_key_here'

def clean_text(text):
    return text.replace('\n', ' ').replace('\r', ' ').strip()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_ocr_results(merge_txt_path):
    """Parse the merge_txt file to map timestamps to concatenated OCR content."""
    timestamp_ocr_map = {}
    with open(merge_txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        current_timestamp = None
        ocr_texts = []

        for line in lines:
            if line.startswith("Timestamp:"):
                if current_timestamp is not None:
                    # Join all OCR texts for the previous timestamp
                    timestamp_ocr_map[current_timestamp] = " ".join(ocr_texts)
                current_timestamp = line.split()[1]
                ocr_texts = []
            elif line.startswith("OCR"):
                # Extract OCR text after the bounding box
                ocr_text = line.split(')', 1)[1].strip()
                ocr_texts.append(ocr_text)
        
        # Don't forget to add the last processed timestamp
        if current_timestamp is not None:
            timestamp_ocr_map[current_timestamp] = " ".join(ocr_texts)

    return timestamp_ocr_map


def voice_gpt(txt_data, text, ppt_image_path):
    headers_voice = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    base64_image = encode_image(ppt_image_path)
    template = '"```json\n[\n    {\n        \"sentence\": \"And this classification is based on the instructions that the processors are executing in parallel.\",\n        \"range\": \"25800-30000\",\n        \"kp_id\": \"KP1, KP2\"\n    }\n]\n```"'
    prompt_text = (
    "This task involves aligning spoken text with knowledge points from a slide. "
    "You have the following knowledge points extracted from the slide: " + txt_data + ". "
    "The spoken text and its time range are provided as follows: " + text + ". "
    "The image attached represents the slide being discussed during the class. "
    "Your task is to match each sentence in the spoken text with the most relevant knowledge point(s) from the slide. "
    "Focus on matching the content of the knowledge points, considering both the text provided and the visual information from the slide image. "
    "Based on the context and the PPT slides, please try to interpret what the teacher intended to convey with each segment of their speech from the teacher's perspective."
    "You need to compare each sentence with every knowledge point and select the one with the highest relevance. Use the content of the 'KP_description' as the primary comparison reference, with other information serving as supplementary knowledge."
    "Ensure that the match is based on the actual content and context of the knowledge points. "
    "Every sentence in the spoken text should be associated with at least one knowledge point (kp_id), or more if applicable. "
    "The output should be in JSON format, strictly following the structure provided in this template: " + template + ". "
    "Ensure consistency in the format, and provide the results in JSON format, with keys for sentence (the spoken sentence), range (time range of the sentence), and kp_id (the corresponding knowledge point ID, e.g.," +template+ ")."
)
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                          "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
                
            }
        ]
    }

    retry_count = 0
    while retry_count < 3:  # Retry up to 3 times
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers_voice, json=payload)
            response.raise_for_status()  # Raise exception for bad response status
            
            if response.status_code == 200:
                return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            retry_count += 1
            time.sleep(10)  # Wait for 10 seconds before retrying

    return {"error": "Failed to get response after multiple retries."}

def convert_json_to_txt_en(file_path):
    # 确定输出文件的目录和文件名
    output_directory = os.path.dirname(file_path)
    base_filename = os.path.basename(file_path).replace('.json', '.txt')
    txt_file_path = os.path.join(output_directory, base_filename)
    
    # 打开并读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    # 提取content字段中的JSON数据
    content_str = json_data['choices'][0]['message']['content']
    
    # 提取JSON字符串部分
    start_index = content_str.find('```json') + len('```json\n')
    end_index = content_str.find('\n```', start_index)
    clean_json_str = content_str[start_index:end_index].strip()
    
    # 尝试将清理后的字符串解析为JSON
    try:
        knowledge_points_data = json.loads(clean_json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return False  # 返回 False 表示解析失败

    # 检查数据是否为列表
    if isinstance(knowledge_points_data, list):
        # 处理多个知识点的情况
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            for item in knowledge_points_data:
                txt_file.write(f"Knowledge_point_id: {item['Knowledge_point_id']}\n")
                txt_file.write(f"Keyword: {clean_text(item['Keyword'])}\n")
                txt_file.write(f"KP_description: {clean_text(item['KP_description'])}\n")
                txt_file.write(f"KP_summary: {clean_text(item['KP_summary'])}\n\n")
   
        print(f"Information saved to {txt_file_path}")
    elif isinstance(knowledge_points_data, dict):
        # 处理单个知识点的情况
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(f"Knowledge_point_id: {knowledge_points_data['Knowledge_point_id']}\n")
            txt_file.write(f"Keyword: {clean_text(knowledge_points_data['Keyword'])}\n")
            txt_file.write(f"KP_description: {clean_text(knowledge_points_data['KP_description'])}\n\n")
            txt_file.write(f"KP_summary: {clean_text(knowledge_points_data['KP_summary'])}\n")
        
        print(f"Information saved to {txt_file_path}")
    else:
        print("JSON data does not contain valid knowledge points.")
    return True  # 返回 True 表示解析成功


def detection_gpt(detection_image_path, original_image_path, knowledge_txt):
    """
    Send an image and knowledge content to OpenAI API and retrieve the knowledge point ID.
    """
    headers_detection = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
 
    base64_image1 = encode_image(detection_image_path)
    base64_image2 = encode_image(original_image_path)
    prompt_text = "Assume you are in a classroom. The global image I provided is a slide from your lecture's PowerPoint presentation, and the partial image is an illustration from that slide. As the teacher...This is the content of the knowledge points: "+ knowledge_txt + " There are Knowledge_point_id, Keyword, KP_description, KP_summary. The image I gave has a global image and a local image, and the partial image is part of the global image. Please analyze and judge which knowledge points the picture belongs to according to the KP_summary and pictures I provide, and only answer the Knowledge_point_id of the knowledge points to which the partial picture belongs, and there is no other content. Especially check if the KP_summary appears on the partial image; the KP_summary should match the text that appears on the image. This image may be related to certain image knowledge points. Examples of outputs are KP1, KP2, and so on. The contents of the key and value must be on the same line."
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": prompt_text
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image1}"
                }
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image2}"
                }
              }
            ]
          }
        ],
        # "max_tokens": 800
    }

    # 设置重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    # 使用会话进行请求，添加重试逻辑
    session = requests.Session()
    session.mount("https://", adapter)

    try:
        response = session.post("https://api.openai.com/v1/chat/completions", headers=headers_detection, json=payload)
        response.raise_for_status()  # 抛出异常如果请求不成功

        time.sleep(3)  # 添加3秒的延迟

        if response.status_code == 200:
            openai_response = response.json()
            knowledge_point_id = openai_response['choices'][0]['message']['content'] if 'content' in openai_response['choices'][0]['message'] else 'No Knowledge Point ID found'
            print("Knowledge_point_id:", knowledge_point_id)
            return knowledge_point_id

    except requests.exceptions.RequestException as e:
        print(f"Failed to process image {detection_image_path}: {e}")

    return None


def generate_json_with_gpt(payload, headers, json_file_path):
    retry_count = 0
    while retry_count < 3:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            if response.status_code == 200:
                openai_response = response.json()
                with open(json_file_path, "w", encoding='utf-8') as json_file:
                    json.dump(openai_response, json_file)
                if not convert_json_to_txt_en(json_file_path):  # 检查是否解析成功
                    raise ValueError("Failed to decode JSON")
                print(f"Saved JSON to {json_file_path}")
                return True  # 如果成功，返回True

        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            print(f"Error processing {json_file_path}: {e}")
            retry_count += 1
            time.sleep(10)

    if retry_count == 3:
        print(f"Failed to process {json_file_path} after {retry_count} retries.")
        return False  # 如果重试失败，返回False



def keypoint_gpt(image_directory, merge_txt, json_directory):
    headers_keypoint = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    os.makedirs(json_directory, exist_ok=True)
    ocr_results = parse_ocr_results(merge_txt)
    example = """```json
[
  {
    "Knowledge_point_id": "KP1",
    "Keyword": "Operating Systems Lecture Information",
    "KP_description": "CS162 Operating Systems and Systems Programming Lecture 11 Scheduling (finished), Deadlock, Address Translation March 5th, 2018 Prof. John Kubiatowicz http://cs162.eecs.Berkeley.edu",
    "KP_summary": "This slide provides information about Lecture 11 of the CS162 course on Operating Systems and Systems Programming, which covers topics including Scheduling, Deadlock, and Address Translation. The lecture took place on March 5th, 2018, and was given by Prof. John Kubiatowicz."
  },
  {
    "Knowledge_point_id": "KP2",
    "Keyword": "Operating Systems Lecture Information",
    "KP_description": "CS162 Operating Systems and Systems Programming Lecture 11 Scheduling (finished), Deadlock, Address Translation March 5th, 2018 Prof. John Kubiatowicz http://cs162.eecs.Berkeley.edu",
    "KP_summary": "This slide provides information about Lecture 11 of the CS162 course on Operating Systems and Systems Programming, which covers topics including Scheduling, Deadlock, and Address Translation. The lecture took place on March 5th, 2018, and was given by Prof. John Kubiatowicz."
  }
]
```"""
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg"):
            timestamp = filename[:-4]
            image_path = os.path.join(image_directory, filename)
            base64_image = encode_image(image_path)
            ocr_content = ocr_results.get(timestamp, "No OCR results found.")

            payload = {
                "model": "gpt-4o-2024-08-06",

                "messages": [
                  {
                    "role": "user",
                    "content": [
                      {
                        "type": "text",
                        "text": f"""
                                Timestamp: {timestamp}.
                                Please analyze the knowledge points depicted in this image, which originates from an educational PPT. 
                                Categorize the entire page into distinct knowledge point areas based solely on the visible content. Ensure that there are no overlapping pixel ranges for each designated area.
                                Assign a unique knowledge point ID to each area, such as KP1, KP2, etc.
                                The 'KP_description' should refer to the text exactly as it would be if OCR were applied, assuming standard text recognition from the image.
                                'KP_summary' should offer a comprehensive explanation of each knowledge point, elaborating in detail based on the visible content and context provided by the image.
                                Ensure that there is no content overlap between different areas.
                                Output the divisions in a structured JSON format with keys including 'Knowledge_point_id', 'Keyword', 'KP_description', and 'KP_summary'.
                                The content for each key should not include line breaks but can be connected using an English comma.
                                Ensure that all outputs are related strictly to the JSON data and that the format is consistent and error-free at all times.
                                This request does not require responding to specific OCR text, but the analysis should be as accurate as if OCR results were provided.
                                Below is an example of the desired output format: {example}
                                """
                      },
                      {
                        "type": "image_url",
                        "image_url": {
                          "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                      }
                    ]
                  }
                ],
            }

            json_file_name = filename[:-4] + ".json"
            json_file_path = os.path.join(json_directory, json_file_name)
            success = generate_json_with_gpt(payload, headers_keypoint, json_file_path)
            if not success:  # 如果生成失败，重新生成该时间戳对应的回答
                generate_json_with_gpt(payload, headers_keypoint, json_file_path)

    print("All images processed.")

# 定义计算KP_id数量的函数
def count_kp_ids(text):
    # 使用正则表达式匹配所有的Knowledge_point_id
    kp_ids = re.findall(r'Knowledge_point_id:', text)
    # 返回匹配到的数量
    return len(kp_ids)


