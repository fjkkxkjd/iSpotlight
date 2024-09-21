import cv2
import re
import random

def parse_timestamp_data(file_path):
    """
    Parses the timestamp data from the input file.
    
    :param file_path: Path to the file containing timestamp and knowledge point data.
    :return: List of dictionaries with timestamp and corresponding areas.
    """
    data = []
    current_timestamp = None
    current_areas = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("Timestamp:"):
                # Append the previous timestamp's data
                if current_timestamp is not None:
                    data.append({"timestamp": current_timestamp, "areas": current_areas})
                # Start a new timestamp section
                current_timestamp = int(line.split(": ")[1].strip())
                current_areas = []
            elif "Knowledge_point_id" in line:
                # Parse area coordinates and knowledge point IDs
                area_part, kp_part = line.split(": Knowledge_point_id: ")
                coords = tuple(map(int, re.findall(r'\d+', area_part)))
                kp_ids = kp_part.strip().split(", ")
                current_areas.append({"coords": coords, "kp_ids": kp_ids})
        # Append the last timestamp's data
        if current_timestamp is not None:
            data.append({"timestamp": current_timestamp, "areas": current_areas})
    return data

def generate_color_for_kp_ids(kp_ids):
    """
    Generates a unique color for each knowledge point ID using a palette of visually appealing colors.
    
    :param kp_ids: List of knowledge point IDs.
    :return: Dictionary mapping knowledge point IDs to colors.
    """
    # Predefined set of visually appealing colors (R, G, B)
    color_palette = [
        (118,218,145),  
        (248,203,127),  
        (248,149,136),  
        (145,146,171),  
        (120,152,225),  
        (239,166,102), 
        (237,221,134),  
        (153,135,206),  
        (99,178,238),  
        (118,218,145)    
    ]
    
    colors = {}
    random.seed(42)  # Seed for consistent color generation across runs
    palette_size = len(color_palette)
    
    for i, kp_id in enumerate(kp_ids):
        # Assign a color from the palette, cycling through if needed
        colors[kp_id] = color_palette[i % palette_size]
        
    return colors

def annotate_video_with_kp(video_path, timestamp_data, output_path):
    """
    Annotates the video based on the timestamp data.
    
    :param video_path: Path to the input video.
    :param timestamp_data: Parsed timestamp data.
    :param output_path: Path to save the annotated video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Get all unique kp_ids for generating colors
    all_kp_ids = set(kp_id for timestamp in timestamp_data for area in timestamp['areas'] for kp_id in area['kp_ids'])
    colors = generate_color_for_kp_ids(all_kp_ids)

    current_areas = []
    current_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Check if we need to update the current areas based on the next timestamp
        if current_index < len(timestamp_data) - 1 and current_time >= timestamp_data[current_index + 1]['timestamp']:
            current_index += 1
            current_areas = timestamp_data[current_index]['areas']

        # Draw the current areas with the appropriate colors
        for area in current_areas:
            for kp_id in area['kp_ids']:
                color = colors[kp_id]
                cv2.rectangle(frame, area['coords'][:2], area['coords'][2:], color, 8)
                cv2.putText(frame, kp_id, (area['coords'][0], area['coords'][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main execution
video_name = "5"
file_path = rf"video_result_data\{video_name}_data\final_match_merge.txt"  # Replace with the path to your data file
video_path = rf"video_data\{video_name}.mp4"  # Replace with the path to your video file
output_path = rf"video_result_data\{video_name}_data\KP_annotated_video.mp4"  # Output path for the annotated video

# Parse timestamp data from file
timestamp_data = parse_timestamp_data(file_path)

# Annotate video with knowledge point areas
annotate_video_with_kp(video_path, timestamp_data, output_path)

print("Video annotation completed.")





