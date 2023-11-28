import os
from faster_whisper_youtube_local_corrected import transcribe_audio
# 假设 audio_path 是您要测试的音频文件路径
video_path = '/Users/ro9air/VScode/Flask_API_TIM/Wav/陳嫺靜 - 有人責備我們不夠深入.mp3'

# 调用 transcribe_audio 函数
transcribed_text = transcribe_audio(video_path)

print("逐字稿:")
print(transcribed_text)
