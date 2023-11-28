
# -*- coding: utf-8 -*-



#! pip install faster-whisper
#! pip install yt-dlp


import sys
import warnings 
from faster_whisper import WhisperModel
from pathlib import Path
#import yt_dlp
import subprocess
import torch
import shutil
import numpy as np
import os
#from IPython.display import display, Markdown, YouTubeVideo
#@markdown ---
model_size = 'medium' #@param ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2']
device_type = "cpu" #@param {type:"string"} ['cuda', 'cpu']
compute_type = "int8" #@param {type:"string"} ['float16', 'int8_float16', 'int8']
#@markdown ---
#@markdown **Run this cell again if you change the model.**

model = WhisperModel(model_size, device=device_type, compute_type=compute_type)

# 设置设备，如有 GPU 则使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print('Using device:', device, file=sys.stderr)
drive_mount_path = Path("/") # Modified for local execution #@param {type:"string"}
drive_path = "/Users/ro9air/VScode/Flask_API_TIM/Wav/劉德音.wav" # Modified for local execution #@param {type:"string"}
#@markdown ---
#@markdown **Run this cell again if you change your Google Drive path.**

drive_whisper_path = Path("/Users/ro9air/VScode/Flask_API_TIM/Server_Recive_file")
drive_whisper_path.mkdir(parents=True, exist_ok=True)

Type = "Google Drive" #@param ['Youtube video or playlist', 'Google Drive']
video_path = "/Users/ro9air/VScode/Flask_API_TIM/Wav/傷心欲絕『下一步絕望』Official Video.mp3" #@param {type:"string"}

language = "zh" #@param ["auto", "en", "zh", "ja", "fr", "de"] {allow-input: true}

initial_prompt = "Please do not translate, only transcription be allowed.  Here are some English words you may need: Cindy. And Chinese words: \u7206\u7834" #@param {type:"string"}
#@markdown ---
#@markdown #### Word-level timestamps
word_level_timestamps = False #@param {type:"boolean"}
#@markdown ---
#@markdown #### VAD filter
vad_filter = True #@param {type:"boolean"}
vad_filter_min_silence_duration_ms = 50 #@param {type:"integer"}
#@markdown ---
#@markdown #### Output(Default is srt, txt if `text_only` be checked )
text_only = True #@param {type:"boolean"}

video_path_local_list = []

def seconds_to_time_format(s):
    # Convert seconds to hours, minutes, seconds, and milliseconds
    hours = s // 3600
    s %= 3600
    minutes = s // 60
    s %= 60
    seconds = s // 1
    milliseconds = round((s % 1) * 1000)

    # Return the formatted string
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def transcribe_audio(video_path, model_size='medium', device_type='cpu', compute_type='int8'):
    if Type == "Google Drive":
        # video_path_drive = drive_mount_path / Path(video_path.lstrip("/"))
        video_path = drive_mount_path / Path(video_path.lstrip("/"))
        if video_path.is_dir():
            for video_path_drive in video_path.glob("**/*"):
                if video_path_drive.is_file():
                    print(f"**{str(video_path_drive)} selected for transcription.**")
                elif video_path_drive.is_dir():
                    print(f"**Subfolders not supported.**")
                else:
                 print(f"**{str(video_path_drive)}  does not exist, skipping.**")
                video_path_local = Path(".").resolve() / (video_path_drive.name)
                shutil.copy(video_path_drive, video_path_local)
                video_path_local_list.append(video_path_local)
        elif video_path.is_file():
            video_path_local = Path(".").resolve() / (video_path.name)
            shutil.copy(video_path, video_path_local)
            video_path_local_list.append(video_path_local)
            print(f"**{str(video_path) } selected for transcription.**")
        else:
            print(f"**{str(video_path)} does not exist, skipping.**")

    else:
        raise(TypeError("Please select supported input type."))

    for video_path_local in video_path_local_list:
        if video_path_local.suffix == ".mp4":
            video_path_local = video_path_local.with_suffix(".wav")
            result  = subprocess.run(["ffmpeg", "-i", str(video_path_local.with_suffix(".mp4")), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(video_path_local)])
                
        segments, info = model.transcribe(str(video_path_local), beam_size=5,
                                        language=None if language == "auto" else language,
                                        initial_prompt=initial_prompt,
                                        word_timestamps=word_level_timestamps,
                                        vad_filter=vad_filter,
                                        vad_parameters=dict(min_silence_duration_ms=vad_filter_min_silence_duration_ms))
            
        #print(f"Detected language '{info.language}' with probability {info.language_probability}")

        ext_name = '.txt' if text_only else ".srt"
        transcript_file_name = video_path_local.stem + ext_name
        sentence_idx = 1
        with open(transcript_file_name, 'w') as f:
            for segment in segments:
                if word_level_timestamps:
                    for word in segment.words:
                        ts_start = seconds_to_time_format(word.start)
                        ts_end = seconds_to_time_format(word.end)
                        #print(f"[{ts_start} --> {ts_end}] {word.word}")
                        if not text_only:
                            f.write(f"{sentence_idx}\n")
                            f.write(f"{ts_start} --> {ts_end}\n")
                            f.write(f"{word.word}\n\n")
                        else:
                            f.write(f"{word.word}")
                        f.write("\n")
                        sentence_idx = sentence_idx + 1
                else:
                    ts_start = seconds_to_time_format(segment.start)
                    ts_end = seconds_to_time_format(segment.end)
                    print(f"[{ts_start} --> {ts_end}] {segment.text}")
                    if not text_only:
                        f.write(f"{sentence_idx}\n")
                        f.write(f"{ts_start} --> {ts_end}\n")
                        f.write(f"{segment.text.strip()}\n\n")
                    else:
                        f.write(f"{segment.text.strip()}\n")
                    sentence_idx = sentence_idx + 1

        def read_transcript(file_path):
            # 确保文件存在
            if not os.path.exists(file_path):
                print("文件不存在:", file_path)
                return ""

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            return content
        
        try:
            print(f"**Transcript file created: {drive_whisper_path / transcript_file_name}**")
            shutil.copy(video_path_local.parent / transcript_file_name,
                            drive_whisper_path / transcript_file_name
                )
            return read_transcript(file_path=drive_whisper_path / transcript_file_name)
        
        except:
            print(f"**Transcript file created: {video_path_local.parent / transcript_file_name}**")
    





