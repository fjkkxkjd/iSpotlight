import re

def merge_detection_ocr(file_path, output_path):
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # 正则表达式匹配时间戳，检测框和OCR框
    timestamp_blocks = re.split(r'Timestamp: \d+', data)[1:]  # 分割数据为每个时间戳的块
    timestamps = re.findall(r'Timestamp: (\d+)', data)  # 时间戳列表

    # 函数：计算矩形面积
    def calculate_area(x1, y1, x2, y2):
        return (x2 - x1) * (y2 - y1)

    # 函数：计算两矩形交集面积
    def intersection_area(box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0
        return calculate_area(x_left, y_top, x_right, y_bottom)

    # 函数：合并两个矩形框
    def merge_boxes(box1, box2):
        return (
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3])
        )

    # 函数：计算两个矩形的高度重叠部分
    def height_overlap(box1, box2):
        y_top = max(box1[1], box2[1])
        y_bottom = min(box1[3], box2[3])
        if y_bottom < y_top:
            return 0
        return y_bottom - y_top

    # 函数：合并同一行中左右两端相差距离小于30个像素并且高度重叠超过60%的OCR框
    def merge_ocr_boxes(ocrs):
        merged_ocrs = []
        skip_indices = set()
        for i, ocr1 in enumerate(ocrs):
            if i in skip_indices:
                continue
            merged_box = ocr1[0]
            text = ocr1[1]
            for j, ocr2 in enumerate(ocrs[i+1:], start=i+1):
                if j in skip_indices:
                    continue
                left_right_distance = min(abs(ocr1[0][2] - ocr2[0][0]), abs(ocr2[0][2] - ocr1[0][0]))
                height_overlap_amount = height_overlap(ocr1[0], ocr2[0])
                if left_right_distance < 30 and height_overlap_amount > 0.6 * min(ocr1[0][3] - ocr1[0][1], ocr2[0][3] - ocr2[0][1]):
                    merged_box = merge_boxes(merged_box, ocr2[0])
                    text += " " + ocr2[1]
                    skip_indices.add(j)
            merged_ocrs.append((merged_box, text))
        return merged_ocrs

    # 存储更新后的结果
    final_results = []

    # 处理每个时间戳块
    for timestamp, block in zip(timestamps, timestamp_blocks):
        detection_texts = re.findall(r'Detection \d+: tensor\(\[(.*?)\]\)', block)
        ocr_texts = re.findall(r'OCR \d+: \((\d+), (\d+)\), \((\d+), (\d+)\), (.+)', block)

        # 解析检测框和OCR框
        detections = [tuple(map(float, d.split(',')[:4])) for d in detection_texts]
        detections = [tuple(map(int, map(round, d))) for d in detections]
        ocrs = [(tuple(map(int, ocr[:-1])), ocr[-1]) for ocr in ocr_texts]

        # 合并同一行中左右两端相差距离小于30个像素并且高度重叠超过60%的OCR框
        ocrs = merge_ocr_boxes(ocrs)

        # 更新检测框和OCR编号
        updated_detections = detections.copy()
        ocrs_to_remove = []

        for o_index, ocr in enumerate(ocrs):
            for d_index, det in enumerate(detections):
                inter_area = intersection_area(det, ocr[0])
                ocr_area = calculate_area(*ocr[0])
                if inter_area >= 0.5 * ocr_area:
                    updated_detections[d_index] = merge_boxes(det, ocr[0])
                    ocrs_to_remove.append(o_index)
                    break

        # 移除被合并的OCR框
        remaining_ocrs = [ocr for index, ocr in enumerate(ocrs) if index not in ocrs_to_remove]

        # 构建结果字符串
        result = f"Timestamp: {timestamp}\n"
        for idx, det in enumerate(updated_detections):
            result += f"Detection {idx+1}: ({det[0]}, {det[1]}, {det[2]}, {det[3]})\n"
        for idx, ocr in enumerate(remaining_ocrs):
            result += f"OCR {idx+1}: ({ocr[0][0]}, {ocr[0][1]}, {ocr[0][2]}, {ocr[0][3]}) {ocr[1]}\n"
        final_results.append(result)

    # 将结果保存到文件
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(final_results))

