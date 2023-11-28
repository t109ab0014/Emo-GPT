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
from faster_whisper_youtube_local_corrected import transcribe_audio
import requests

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
            '無聊': {'x': -0.1, 'y': -0.8, 'z': 0.4},
            '悲傷': {'x': -1.0, 'y': -1.0, 'z': 0.8}
        }
        
        if translated_emotion in emotion_xyz:
            emotion_coordinates = emotion_xyz[translated_emotion]
            print(f"Emotion coordinates: {emotion_coordinates}")  # 调试用
        else:
            print(f"Emotion {translated_emotion} not found in emotion_xyz dictionary")  # 调试用
            emotion_coordinates = {'x': 1, 'y': 1, 'z': 1} #不在情緒字典中的情緒，隨機生成坐標
        emotion_analysis = f"情緒分析結果：{translated_emotion}"

        # 调用 transcribe_audio 函数
        transcribed_text = transcribe_audio(file_path)
        # 呼叫 GPT API 進行總結
        summary = call_gpt_chat_api(transcribed_text,emotion_analysis)
        print(f"Summary: {summary}")  # 调试用

        # 更新global_data
        generate_data(emotion_coordinates)

        return render_template('index.html', emotion=translated_emotion,transcription=transcribed_text,summary=summary)
    
    else:
        return 'No file uploaded', 400
    
import requests

def call_gpt_chat_api(dialogue_content, emotion_analysis):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
        "OpenAI-Organization": "org-Kuc3r0VCGGBiFP9PsJjowlED"
    }
    messages = [
        {"role": "system", "content": "模型任務：1. 分析對話內容，識別是否有符合詐欺模式的跡象。2. 根據情緒分析結果，評估對話中的情緒指標是否與詐欺行為相關。3. 綜合上述分析結果，根據風險導流SOP決定是否需要延後、轉交或拒絕請求。只需要模型輸出：1. [分析結果：詐欺模式#X（若有符合的模式），或「無明顯詐欺模式」] <br> 2. [風險導流決策：「請延後或轉交其他部門」或「請繼續處理」]"},
        {"role": "system", "content": f"對話內容：{dialogue_content}"},
        {"role": "system", "content": f"情緒分析結果：{emotion_analysis}"},
        {"role": "system", "content": f"詐欺模式描述：{fraud_patterns}"},
        {"role": "system", "content": f"風險導流SOP：{risk_flow_SOP}"}
    ]

    data = {
        "model": "gpt-4-1106-preview",  # 指定模型名稱
        "messages": messages,
        "max_tokens": 1000  # 您可能需要更多的 tokens 來處理這些資訊
    }

    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    print(response_data)  # 調試用

    # 從響應中提取分析結果
    if 'choices' in response_data and len(response_data['choices']) > 0 and 'message' in response_data['choices'][0]:
        analysis_result = response_data['choices'][0]['message']['content'].strip()
    else:
        analysis_result = "無法生成分析結果"

    return analysis_result

# 範例參數，您需要根據實際情況填充這些變量
fraud_patterns = """全局欺騙模式：這是當兩個算法檢測到高機率的謊言和極度的壓力時的情況，不受模式或用戶偏好的影響，始終適用。

    欺騙模式#1 - 攻擊性謊言：當人們感覺到強烈的緊張和專注時，可能會說出攻擊性的謊言。

    欺騙模式#2 - 矛盾衝突：當人們感覺到異常的興奮和極度的邏輯或認知壓力時，可能會有欺騙的情況。

    欺騙模式#3 - 明確否認：當人們在直接回應（例如“不，我沒偷包包”）時感覺到極度壓力和高度矛盾，可能是欺騙的警告。

    欺騙模式#4 - 尷尬掩蓋：當人們在不應感到尷尬的情況下感到尷尬時，可能會說謊。

    欺騙模式#5 - 警覺避談：當人們處於極度警覺和低思考水平時，可能是欺騙的跡象。

    欺騙模式#6 - 猶豫不決：當人們在是否“說或停止”（S.O.S.）上感到極度的矛盾時，可能是欺騙的跡象。

    欺騙模式#7 - 異常興奮：當人們感到極度不警覺和非常興奮時，可能是因為他們不習慣說謊，或者只是為了“樂趣”而說謊。

    欺騙模式#8 - 邏輯漏洞：當人們在回答時遇到邏輯問題時，可能是欺騙的跡象。
    欺騙模式#9 - 隱藏真相：當看起來像正常人，但在邊緣情況下可能會說謊時，可以使用此模式。"""
risk_flow_SOP = """graph TB
    Start(開始) --> Question1[您能否要求申請人進一步詳述請求]
    Question1 -->|否| Defer1[延後或轉交請求]
    Question1 -->|是| Question2[您了解他的請求嗎]
    Question2 -->|否| Defer2[延後或轉交請求]
    Question2 -->|是| Question3[您能夠執行或提供請求的內容嗎]
    Question3 -->|否| Defer3[延後或轉交請求]
    Question3 -->|是| Question4[您有執行該請求的權限嗎]
    Question4 -->|否| Defer4[延後或轉交請求]
    Question4 -->|是| RefusalBlock{{條件符合區塊1}}
    RefusalBlock -->|有| Defer5[延後或轉交請求]
    RefusalBlock -->|無| Question5[申請者的身分可以被驗證嗎]
    Question5 -->|否| Defer6[延後或轉交請求]
    Question5 -->|是| VerificationBlock{{條件符合區塊2}}
    VerificationBlock -->|3個或更多| Perform[執行請求]
    VerificationBlock -->|1-2個| Defer7[延後或轉交請求]
    VerificationBlock -->|無| Defer8[延後或轉交請求]

    class RefusalBlock,VerificationBlock fill:#ffdddd,stroke:#333,stroke-width:4px;
    subgraph "條件符合區塊1"
    RefusalBlock[請問有符合以下拒絕條件嗎<br>1.有行政原因拒絕嗎<br>2.有程序上的原因拒絕嗎<br>3.這是一種不尋常或新型的請求嗎<br>4.有其他原因拒絕嗎]
    end

    subgraph "條件符合區塊2"
    VerificationBlock[請問申請者符合以下幾個條件<br>1.申請人的權限級別可以被驗證嗎<br>2.申請人的信用可以被驗證嗎<br>3.您之前與申請人有互動嗎<br>4.您知道申請人活著嗎]
    end"""




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
        #generate_data(reset=True)  # 重置global_data
        send_data_to_client()  # 新數據準備好後，通知客戶端更新數據

def computeZ(x, y):
    data = global_data  # 使用全局變量

    sum = 0
    for point in data:
        gaussian = point['z'] * math.exp(-((x - point['x']) ** 2 / (2 * 0.25 ** 2) + (y - point['y']) ** 2 / (2 * 0.25 ** 2)))
        sum += gaussian

    return sum

def generate_data(new_emotion_data=None, reset=False):
    global global_data
    if reset:
        global_data = [{'x': 0, 'y': 0, 'z': 0}]
        print('global_data reset')  # 调试用
    elif new_emotion_data:
        global_data.append(new_emotion_data)
        print(f'global_data updated: {new_emotion_data}')  # 调试用


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


