# -*- coding: UTF-8 -*-
from TIM import TIMNet, TIM_Net, WeightLayer, Temporal_Aware_Block, SpatialDropout, Chomp1d 
from flask_cors import CORS
from flask_socketio import SocketIO
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import config
import librosa
import os
from flask import Flask, request, jsonify
import time,math,random,threading,os,re
from tqdm import tqdm


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)



# 檢查 "/Users/ro9air/VScode/Flask/Server_Recive_file" 目錄是否存在，如果不存在，創建它
if not os.path.exists(config.AUDIO_SAVE_DIR):
    os.makedirs(config.AUDIO_SAVE_DIR)


# 這個函式可以用來讀取單個音頻文件並計算它的MFCC
def get_mfcc(filename, sr=22050, duration=4, framelength=0.05):
    data, sr = librosa.load(filename, sr=sr)
    time = librosa.get_duration(y=data, sr=sr)
    if time > duration:
        data = data[0:int(sr * duration)]
    else:
        padding_len = int(sr * duration - len(data))
        data = np.hstack([data, np.zeros(padding_len)])
    framesize = int(framelength * sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13, n_fft=framesize)
    mfcc = mfcc.T
    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    mfcc_acc = librosa.feature.delta(mfcc_delta, width=3)
    mfcc = np.hstack([mfcc, mfcc_delta, mfcc_acc])
    return mfcc

# 這個函式可以用來讀取多個音頻文件並計算它們的MFCC
def get_mfccs(wav_files: list, sr=22050, duration=4, framelength=0.05):
    print("正在計算MFCC...")
    mfccs = get_mfcc(wav_files[0], sr=sr, duration=duration, framelength=framelength)
    size = mfccs.shape
    for it in tqdm(wav_files[1:]):
        mfcc = get_mfcc(it, sr=sr, duration=duration, framelength=framelength)
        mfccs = np.vstack((mfccs, mfcc))
    mfccs = mfccs.reshape(-1, size[0], size[1])
    return mfccs

# 函數用於從資料夾讀取音訊並返回其MFCC特徵
def load_and_preprocess_from_folder(folder_path):
    wav_files = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)]
    mfcc_features = get_mfccs(wav_files)  # 假設get_mfccs函數會返回一個NumPy數組
    return mfcc_features


def load_model(model_path):
    # 載入模型
    model_path = config.MODEL_PATH
    # 使用這個代碼行來加載整個模型
    model = torch.load(model_path, map_location='cpu') # weight of size [128, 39, 1]

    # 評估模式
    model.eval()
    model.cpu()  
    return model

def predict_emotion(model, audio_file_path):
    emotion_labels = ['anger', 'boredom', 'disgust', 'fear', 'happy', 'neutral', 'sad']

    # 使用單一音訊檔案計算 MFCC
    x = get_mfcc(audio_file_path)
    
    # 添加一個新的軸以匹配模型輸入
    x = np.expand_dims(x, axis=0)
    
    # 轉換 x 的形狀（如果需要）
    x = np.transpose(x, (0, 2, 1))

    # 使用模型進行情緒分類
    with torch.no_grad():
        predictions = model(torch.tensor(x, dtype=torch.float32))
        
        # 找到預測張量中的最大值的索引
        _, predicted_label_index = torch.max(predictions, 1)
        
        # 使用索引找到對應的情緒標籤
        predicted_emotion = emotion_labels[predicted_label_index.item()]
        
        return predicted_emotion

# 全局變量來存儲模型
model = load_model(config.MODEL_PATH)

def translate_emotion(english_emotion):
    print(f'Translating: {english_emotion}')  # 新添加的列印語句
    translation_dict = {
        'anger': '憤怒',
        'disgust': '厭惡',
        'fear': '恐懼',
        'happy': '快樂',
        'neutral': '中立',
        'boredom': '無聊',
        'sad': '悲傷'
    }
    translated_emotion = translation_dict.get(english_emotion, '未知情緒')  # 返回對應的中文情緒，如果沒有找到，返回 '未知情緒'
    print(f'Translated to: {translated_emotion}')  # 新添加的列印語句
    return translated_emotion

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    files = request.files.getlist('file')
    emotions = []

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400
        audio_file_path = os.path.join(config.AUDIO_SAVE_DIR, file.filename)  # type: ignore
        file.save(audio_file_path)
        predicted_emotion = predict_emotion(model, audio_file_path)
        print(f'Predicted emotion: {predicted_emotion}') 
        #os.remove(audio_file_path)  # 選項：刪除音訊檔案
        
        translated_emotion = translate_emotion(predicted_emotion)
        emotions.append({'filename': file.filename, 'emotion': translated_emotion})

    return jsonify({'results': emotions})

@app.route('/input', methods=['GET'])
def upload_page():
    return render_template('input.html')





@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and isinstance(file.filename, str):
        safe_filename = re.sub('[^a-zA-Z0-9.]+', '_', file.filename)
        file_path = os.path.join('/Users/ro9air/VScode/Flask_API_TIM/Server_Recive_file', safe_filename) # type: ignore
        file.save(file_path)
        
        # Call your SER API here
        predicted_emotion = predict_emotion(model, file_path)
        translated_emotion = translate_emotion(predicted_emotion)
        print(f"Translated emotion: {translated_emotion}")  # 调试用

        emotion_xyz = {
            '憤怒': {'x': -1, 'y': 1, 'z': 0.7},
            '厭惡': {'x': -0.8, 'y':-0.2 , 'z': 0.4},
            '恐懼': {'x': -1, 'y': 0.5 , 'z': 0.9},
            '快樂': {'x': 1, 'y': 0.7, 'z': 0.9},
            '中立': {'x': 0, 'y': 0, 'z': 0.5},
            '無聊': {'x': -0.5, 'y': -1, 'z': 0.4},
            '悲傷': {'x': -1.0, 'y': -1.0, 'z': 0.8}
        }
        
        if translated_emotion in emotion_xyz:
            emotion_coordinates = emotion_xyz[translated_emotion]
        else:
            print(f"Emotion {translated_emotion} not found in emotion_xyz dictionary")  # 调试用
            emotion_coordinates = {'x': 1, 'y': 1, 'z': 1}

        # 更新global_data
        generate_data(emotion_coordinates)

        return render_template('index.html', emotion=translated_emotion)
    else:
        return 'No file uploaded', 400


@app.route('/')
def root():
    return redirect(url_for('upload_page'))


global_data = []

def send_data_to_client():
    data = generate_data()
    socketio.emit('update_data', data)

def update_data():
    while True:
        time.sleep(3)  # 每3秒更新一次數據
        send_data_to_client()  # 新數據準備好後，通知客戶端

def computeZ(x, y):
    data = global_data  # 使用全局變量

    sum = 0
    for point in data:
        gaussian = point['z'] * math.exp(-((x - point['x']) ** 2 / (2 * 0.25 ** 2) + (y - point['y']) ** 2 / (2 * 0.25 ** 2)))
        sum += gaussian

    return sum

def generate_data(new_emotion_data=None):
    global global_data
    if new_emotion_data:
        global_data = [new_emotion_data]
    else:
        global_data = [
            {'x': 0, 'y': 0, 'z': 0},
        ]

    data = []
    for x in range(-150, 151, 5):  # 修改步長以匹配原始步長 0.05
        for y in range(-150, 151, 5):
            z = computeZ(x/100, y/100)  # 調用 computeZ 函數
            data.append([x/100, y/100, z])
    for y in range(-150, 151, 5):
        for x in range(-150, 151, 5):
            z = computeZ(x/100, y/100)  # 調用 computeZ 函數
            data.append([x/100, y/100, z])
    return data




if __name__ == "__main__":
    data_thread = threading.Thread(target=update_data)
    data_thread.start()
    socketio.run(app, host='0.0.0.0', debug=True, port=50500) # type: ignore


