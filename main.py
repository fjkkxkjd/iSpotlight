# Imports and Dependencies
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTConfig, ViTModel
from pix2text import Pix2Text
from detect_dual import run
from gpt import keypoint_gpt, voice_gpt
from overlap import merge_detection_ocr
from kp_match_merge import kp_match_data
from voice_to_txt import transcribe_video_to_tsv
from voice import process_result_json_folder, process_data_and_prepare_for_voice_gpt
from voice_match import process_voice_and_final_data

# Parameter Settings
CONFIG_PATH = "weights/config.json"
WEIGHTS_PATH = "weights/best_difference_vit_mlff_model.pth"
PIX2TEXT_DEVICE = 'cpu'
PIX2TEXT_RESIZE = 768
DETECTION_WEIGHTS = r'weights\yolo_best.pt'
DETECTION_DATA = r'data\video.yaml'
DETECTION_IMGSZ = (640, 640)
DETECTION_CONF_THRES = 0.40
DETECTION_IOU_THRES = 0.01
VIDEO_NAME = "5"
VIDEO_PATH = rf"video_data\{VIDEO_NAME}.mp4"
OUTPUT_VIDEO = rf"video_result_data\{VIDEO_NAME}_data\{VIDEO_NAME}_out.mp4"
OUTPUT_TEXT = rf'video_result_data\{VIDEO_NAME}_data\recognized_text.txt'
TIMESTAMPS_FILE = rf'video_result_data\{VIDEO_NAME}_data\timestamps.txt'
OBJECT_FILE = rf"video_result_data\{VIDEO_NAME}_data\object_frames"
JSON_DIRECTORY = rf"video_result_data\{VIDEO_NAME}_data\json_files"
MERGE_PATH = rf'video_result_data\{VIDEO_NAME}_data\merge.txt'
FINAL_MATCH_PATH = rf'video_result_data\{VIDEO_NAME}_data\final_match_merge.txt'
TSV_FILE_PATH = rf"video_result_data\{VIDEO_NAME}_data\{VIDEO_NAME}.tsv"
RESULT_JSON_FILES_DIR = rf'video_result_data\{VIDEO_NAME}_data\result_json_files'
VOICE_KP_MATCH_FILE = rf'video_result_data\{VIDEO_NAME}_data\voice_match_final.txt'
VOICE_JSON_FILES_DIR = rf'video_result_data\{VIDEO_NAME}_data\voice_json_files'
OUTPUT_FRAMES_FOLDER = rf"video_result_data\{VIDEO_NAME}_data\original_frames"
ALIGNMENT_FILE_PATH = rf"video_result_data\{VIDEO_NAME}_data\merged_results.json"
WHISPER_MODEL_PATH = "weights/large-v3.pt"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cuda:0"
DIFFERENCE_THRESHOLD = 0.005

# Utility Functions
def ensure_folder_exists(folder):
    """Creates a folder if it does not exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder {folder} created")

def load_pretrained_weights(model, path):
    """Loads pretrained weights into a model."""
    model.load_state_dict(torch.load(path))

# Image Processing Functions
def remove_black_borders(image_path, output_path):
    """Removes black borders from an image."""
    with Image.open(image_path) as img:
        gray = img.convert("L")
        bbox = gray.point(lambda x: 255 if x > 35 else 0).getbbox()
        if bbox:
            img_cropped = img.crop(bbox)
            img_cropped.save(output_path)
        else:
            print("No black borders detected for cropping.")

def recognize_formula(frame):
    """Recognizes formulas in an image using Pix2Text."""
    p2t = Pix2Text.from_config(device=PIX2TEXT_DEVICE)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = p2t.recognize_text_formula(img, resized_shape=PIX2TEXT_RESIZE, return_text=False)
    return results

def preprocess_image(frame, device=DEVICE):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    return image_tensor


def predict_similarity(model, image1, image2, threshold, device=DEVICE):
    """Predicts similarity between two frames using a given model."""
    model.eval()
    diff = torch.abs(image1 - image2).sum(dim=(1, 2, 3))  
    diff = diff/(224*224)
    if  (diff > threshold):
        with torch.no_grad():
            similarity = model(image1, image2)
            flag = similarity.item() > 0.6
    else:
        flag = 1

    return flag


class ViTWithMLFF(nn.Module):
    def __init__(self, config):
        super(ViTWithMLFF, self).__init__()
        self.vit = ViTModel(config)
        self.conv1 = nn.Conv2d(768, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(768, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(768, 256, kernel_size=1)
        self.final_conv = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, input1, input2):
        diff = torch.abs(input1 - input2)
        outputs = self.vit(diff, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        feature1 = self.conv1(hidden_states[5][:, 1:, :].transpose(1, 2).view(-1, 768, 14, 14))
        feature2 = self.conv2(hidden_states[7][:, 1:, :].transpose(1, 2).view(-1, 768, 14, 14))
        feature3 = self.conv3(hidden_states[-1][:, 1:, :].transpose(1, 2).view(-1, 768, 14, 14))
        combined_features = feature1 + feature2 + feature3
        output = self.final_conv(combined_features)
        return torch.sigmoid(output.mean([2, 3]))

# Video Processing Functions
def save_frame(frame, current_index, fps, output_frames_folder):
    """Saves a frame from the video."""
    timestamp_ms = current_index * 1000 // fps
    full_path = os.path.join(output_frames_folder, f"{timestamp_ms}.jpg")
    ensure_folder_exists(output_frames_folder)
    cv2.imwrite(full_path, frame)

def process_frame(frame, file, current_index, fps, object_file, output_frames_folder):
    """Processes a frame for object detection and OCR."""
    timestamp_ms = current_index * 1000 // fps
    timestamp_filename = f"{timestamp_ms}.jpg"
    cv2.imwrite(timestamp_filename, frame)

    detections = run(weights=DETECTION_WEIGHTS, source= timestamp_filename,
                     data=DETECTION_DATA, imgsz=DETECTION_IMGSZ, conf_thres=DETECTION_CONF_THRES,
                     iou_thres=DETECTION_IOU_THRES, device=DEVICE)
    
    os.remove(timestamp_filename)  # Remove the temporary image file
    if detections is not None:
        for obj_index, obj_det in enumerate(detections[0]):
            for det_index, det in enumerate(obj_det):
                file.write(f"Detection {det_index + 1}: {det}\n")
                det = det.tolist()
                top_left = (int(det[0]), int(det[1]))
                bottom_right = (int(det[2]), int(det[3]))
                obj_frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                obj_filename = f"{timestamp_ms}_detection{det_index + 1}.jpg"
                cv2.imwrite(os.path.join(object_file, obj_filename), obj_frame)
    else:
        file.write("No objects detected\n")
    
    ocr_results = recognize_formula(frame)
    ocr_count = 1
    for detection in ocr_results:
        position = detection['position']
        text = detection['text']
        top_left = (int(position[0][0]), int(position[0][1]))
        bottom_right = (int(position[2][0]), int(position[2][1]))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        file.write(f"OCR {ocr_count}: {top_left}, {bottom_right}, {text}\n")
        print(f"OCR {ocr_count}: {top_left}, {bottom_right}, {text}\n")
        ocr_count += 1
    file.write("\n")
    print("Frame processing complete.")


def recognize_and_draw_boxes_on_video(video_path, output_text, output_frames_folder, difference_threshold):
    """Recognizes and draws boxes on video frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    config = ViTConfig.from_pretrained(CONFIG_PATH)
    model = ViTWithMLFF(config).to(DEVICE)
    load_pretrained_weights(model, WEIGHTS_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, prev_frame = cap.read()
    prev_frame_tensor = preprocess_image(prev_frame)
    current_index = 0
    timestamp_get = []

    with open(output_text, 'w', encoding='utf-8') as file:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            curr_frame_tensor = preprocess_image(frame)
            if current_index % 25 == 0:
                # print(current_index)
                similarity = predict_similarity(model, curr_frame_tensor, prev_frame_tensor, difference_threshold)
                timestamp_ms = current_index * 1000 // fps
                if not similarity or current_index == 0:
                    file.write(f"Timestamp: {timestamp_ms}\n")
                    save_frame(frame, current_index, fps, OUTPUT_FRAMES_FOLDER)
                    process_frame(frame, file, current_index, fps, OBJECT_FILE, output_frames_folder)
                    prev_frame_tensor = curr_frame_tensor
                    prev_frame = frame.copy()
                    timestamp_get.append(timestamp_ms)
            current_index += 1

    with open(TIMESTAMPS_FILE, 'w', encoding='utf-8') as tf:
        for timestamp_ms in timestamp_get:
            tf.write(f"{timestamp_ms}\n")
    cap.release()

# Main Execution Flow
def run_process(func, *args, **kwargs):
    """Helper function to run each process step-by-step with error handling."""
    try:
        func(*args, **kwargs)
        print(f"Successfully completed: {func.__name__}")
    except Exception as e:
        print(f"Error occurred in {func.__name__}: {e}")
        exit(1)  # Exit the program if any step fails

def main():
    # Ensure all necessary folders exist
    paths = [
        OUTPUT_FRAMES_FOLDER,
        OBJECT_FILE,
        JSON_DIRECTORY,
        VOICE_JSON_FILES_DIR,
        RESULT_JSON_FILES_DIR,
        OUTPUT_TEXT,
        MERGE_PATH,
        TIMESTAMPS_FILE,
        VOICE_KP_MATCH_FILE,
        FINAL_MATCH_PATH
    ]

    for path in paths:
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        ensure_folder_exists(folder)

    # Run processes sequentially with error handling
    run_process(transcribe_video_to_tsv, video_path=VIDEO_PATH, model_path=WHISPER_MODEL_PATH, output_path=TSV_FILE_PATH, device=DEVICE)
    run_process(recognize_and_draw_boxes_on_video, VIDEO_PATH, OUTPUT_TEXT, OUTPUT_FRAMES_FOLDER, DIFFERENCE_THRESHOLD)
    run_process(merge_detection_ocr, OUTPUT_TEXT, MERGE_PATH)
    run_process(keypoint_gpt, OUTPUT_FRAMES_FOLDER, MERGE_PATH, JSON_DIRECTORY)
    run_process(kp_match_data, MERGE_PATH, JSON_DIRECTORY, OUTPUT_FRAMES_FOLDER, OBJECT_FILE, FINAL_MATCH_PATH)
    run_process(process_data_and_prepare_for_voice_gpt, TSV_FILE_PATH, TIMESTAMPS_FILE, VOICE_JSON_FILES_DIR, RESULT_JSON_FILES_DIR, voice_gpt, JSON_DIRECTORY, VIDEO_PATH, OUTPUT_FRAMES_FOLDER)
    run_process(process_result_json_folder, RESULT_JSON_FILES_DIR, VOICE_KP_MATCH_FILE)
    run_process(process_voice_and_final_data, VOICE_KP_MATCH_FILE, FINAL_MATCH_PATH, ALIGNMENT_FILE_PATH)

if __name__ == "__main__":
    main()
