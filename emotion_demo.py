#! python3.7
import torch
import numpy as np
import librosa
import os
from tqdm import tqdm
import config
# 確保您的模型和函數可以從這裡導入
from TIM import TIMNet, TIM_Net, WeightLayer, Temporal_Aware_Block, SpatialDropout, Chomp1d 
import soundfile as sf

import argparse
import io
import os
import speech_recognition as sr
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform


# 這個函式可以用來讀取單個音頻文件並計算它的MFCC
def get_mfcc(audio_data, sr=22050, duration=4, framelength=0.05):
    target_length = int(sr * duration)
    num_channels = audio_data.shape[1] if audio_data.ndim > 1 else 1
    current_length = audio_data.shape[0]

    # 如果音频长度小于目标长度，则填充
    if current_length < target_length:
        padding_length = target_length - current_length
        padding = np.zeros((padding_length, num_channels)) if num_channels > 1 else np.zeros(padding_length)
        audio_data = np.vstack([audio_data, padding]) if num_channels > 1 else np.hstack([audio_data, padding])

    # 如果音频长度大于目标长度，则截断
    elif current_length > target_length:
        audio_data = audio_data[:target_length]

    # 确保音频数据是一维的
    audio_data_flattened = audio_data.flatten() if num_channels > 1 else audio_data

    # 计算 MFCC
    framesize = int(framelength * sr)
    mfcc = librosa.feature.mfcc(y=audio_data_flattened, sr=sr, n_mfcc=13, n_fft=framesize)
    mfcc = mfcc.T
    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    mfcc_acc = librosa.feature.delta(mfcc_delta, width=3)
    mfcc = np.hstack([mfcc, mfcc_delta, mfcc_acc])
    return mfcc


def load_model(model_path):
    # 載入模型
    model_path = config.MODEL_PATH
    # 使用這個代碼行來加載整個模型
    model = torch.load(model_path, map_location='cpu') # weight of size [128, 39, 1]

    # 評估模式
    model.eval()
    model.cpu()  
    return model

def predict_emotion(model, audio_data):
    emotion_labels = ['anger', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']

    # 使用音频数据计算 MFCC
    x = get_mfcc(audio_data)
    
    # 添加一个新的轴以匹配模型输入
    x = np.expand_dims(x, axis=0)
    
    # 转换 x 的形状（如果需要）
    x = np.transpose(x, (0, 2, 1))

    # 使用模型进行情绪分类
    with torch.no_grad():
        predictions = model(torch.tensor(x, dtype=torch.float32))
        
        # 找到预测张量中的最大值的索引
        _, predicted_label_index = torch.max(predictions, 1)
        
        # 使用索引找到对应的情绪标签
        predicted_emotion = emotion_labels[predicted_label_index.item()]
        
        return predicted_emotion



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # 全局變量來存儲情緒分析模型
    emotion_model = load_model(config.MODEL_PATH)

    # (錄音設置，參考 transcribe_demo.py)
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False


    # (錄音循環開始)
    print("Model loaded. Start speaking...\n")
    while True:
        try:
            with sr.Microphone(sample_rate=16000) as source:
                recorder.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = recorder.listen(source, timeout=args.record_timeout)

                # 将 BytesIO 对象转换为 NumPy 数组
                with io.BytesIO(audio.get_wav_data()) as wav_file:
                    with sf.SoundFile(wav_file) as sound_file:
                        audio_data = sound_file.read(dtype="float32")
                        samplerate = sound_file.samplerate

                mfcc = get_mfcc(audio_data, sr=samplerate)

                # 使用情緒分析模型進行預測
                predicted_emotion = predict_emotion(emotion_model, mfcc)
                print(f"Predicted Emotion: {predicted_emotion}")

                # 限制 CPU 使用
                sleep(0.25)

        except KeyboardInterrupt:
            print("Exiting...")
            break

if __name__ == "__main__":
    main()