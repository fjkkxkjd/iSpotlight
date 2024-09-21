import cv2
import numpy as np
import re
import os
import glob

def process_video(video_folder, gaze_data, frame_data):
    """
    Processes each video based on gaze data, drawing gaze points on the frames.
    """
    for clipname, points in gaze_data.items():
        video_path, output_path, log_path = get_paths(video_folder, clipname)
        
        # Check if the video file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue

        # Set up video writer
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        # Process frames and log gaze points
        write_video_frames(cap, out, points, frame_data, fps, log_path)

        cap.release()
        out.release()
        print(f"Video processing complete for: {clipname}")

def get_paths(video_folder, clipname):
    """
    Returns the paths for the video file, output file, and log file.
    """
    video_path = os.path.join(video_folder, f"{clipname}.mp4")
    output_path = os.path.join(video_folder, f"output_{clipname}.avi")
    log_path = os.path.join(video_folder, f"gaze_log_{clipname}.txt")
    return video_path, output_path, log_path

def write_video_frames(cap, out, points, frame_data, fps, log_path):
    """
    Writes processed frames to the output video and logs gaze points.
    """
    frame_count = 0
    success, frame = cap.read()

    with open(log_path, 'w') as log_file:
        while success:
            current_time = frame_count / fps
            if current_time in points:
                draw_gaze_points(frame, points[current_time], frame_data, cap, log_file, current_time)

            out.write(frame)
            success, frame = cap.read()
            frame_count += 1

def draw_gaze_points(frame, point_list, frame_data, cap, log_file, current_time):
    """
    Draws gaze points on the frame and logs their positions.
    """
    for point in point_list:
        x_pixel = int((point[0] - frame_data['position'][0] + frame_data['width'] / 2) * cap.get(3) / frame_data['width'])
        y_pixel = int((frame_data['height'] / 2 - point[1] + frame_data['position'][1]) * cap.get(4) / frame_data['height'])
        cv2.circle(frame, (x_pixel, y_pixel), 10, (0, 0, 255), -1)
        log_file.write(f"Time: {current_time:.2f}s, Pixel Position: ({x_pixel}, {y_pixel})\n")

def load_gaze_data(file_path):
    """
    Loads gaze data and frame information from a file.
    """
    gaze_data = {}
    frame_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # Parse frame information
        frame_data = parse_frame_data(content)

        # Parse gaze data
        gaze_data = parse_gaze_data(content)

    return gaze_data, frame_data

def parse_frame_data(content):
    """
    Parses frame information such as position, width, and height.
    """
    frame_data = {}
    position_match = re.search(r'rawImageWorldPosition: \(([\d.]+), ([\d.]+), ([\d.]+)\)', content)
    width_match = re.search(r'rawImageworldWidth: ([\d.]+)', content)
    height_match = re.search(r'rawImageworldHeight: ([\d.]+)', content)

    if position_match:
        frame_data['position'] = (float(position_match.group(1)), float(position_match.group(2)), float(position_match.group(3)))
    if width_match:
        frame_data['width'] = float(width_match.group(1))
    if height_match:
        frame_data['height'] = float(height_match.group(1))

    return frame_data

def parse_gaze_data(content):
    """
    Parses gaze data from the content, extracting times, clip names, and positions.
    """
    gaze_data = {}
    gaze_matches = re.finditer(r'CurrentTime: ([\d.]+);Clipname: ([\w_]+)\nposition: \(([\d.]+), ([\d.]+), ([\d.]+)\)', content)
    for match in gaze_matches:
        time = float(match.group(1))
        clipname = match.group(2)
        position = (float(match.group(3)), float(match.group(4)))
        if clipname not in gaze_data:
            gaze_data[clipname] = {}
        gaze_data[clipname].setdefault(time, []).append(position)

    return gaze_data


def find_eye_tracking_file(directory):
    pattern = os.path.join(directory, "EyeTracking_*.txt")
    
    matching_files = glob.glob(pattern)
    
    if matching_files:
        print(f"Found file: {matching_files[0]}")
        return matching_files[0]
    else:
        print("No matching EyeTracking file found.")
        return None

# Example usage
USER_ID = "3"
video_folder = r"video_data"
directory = rf"eye_tracking_data\user_{USER_ID}"
file_path = find_eye_tracking_file(directory)

if file_path:
    gaze_data, frame_data = load_gaze_data(file_path)
    process_video(video_folder, gaze_data, frame_data)
else:
    print("No EyeTracking file found, please check the directory.")