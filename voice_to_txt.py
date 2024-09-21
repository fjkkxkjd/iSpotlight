import subprocess
import whisper
import csv
import os

def extract_audio_with_ffmpeg(video_path, audio_path):
    """
    使用 ffmpeg 从视频文件中提取音频。
    """
    command = [
        "ffmpeg",
        "-i", video_path,  # 输入视频文件路径
        "-vn",             # 不处理视频流
        "-acodec", "pcm_s16le",  # 设置音频编码为 PCM 16位小端
        "-ar", "16000",    # 设置音频采样率为 16000 Hz
        "-ac", "1",        # 设置音频为单声道
        audio_path         # 输出音频文件路径
    ]
    subprocess.run(command, check=True)

def transcribe_video_to_tsv(video_path, model_path, output_path, device="cuda"):
    """
    将视频文件中的音频转录为文字，并保存到指定的输出文件夹中。
    """
    # 从视频文件路径中提取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(os.path.dirname(video_path), base_name + ".wav")

    # 使用 ffmpeg 从视频中提取音频
    extract_audio_with_ffmpeg(video_path, audio_path)

    # 加载 Whisper 模型
    model = whisper.load_model(model_path, device=device)

    # 转录音频
    result = model.transcribe(audio_path, verbose=False)

    # 打开一个文件以写入 TSV
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("start\tend\ttext\n")  # 写入 TSV 头部
        for segment in result['segments']:
            start = int(segment['start']*1000)
            end = int(segment['end']*1000)
            text = segment['text'].replace('\n', ' ')  # 确保文本中没有换行符
            file.write(f"{start}\t{end}\t{text}\n")






