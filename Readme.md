```markdown
# Emo-GPT - 語音情緒辨識API

## 專案概述
Emo-GPT是一個基於PyTorch的語音情緒辨識系統，通過分析音訊檔案，判斷說話者的情緒。

## 快速開始

### 環境設置
- Python 3.8+
- PyTorch 1.7+
- Flask
- librosa

```bash
pip install torch flask librosa
```

### 運行服務器

1. 克隆此存儲庫至您的本地機器。
2. 在終端中切換至專案目錄。
3. 運行以下指令啟動服務器：
```bash
python integrated_app.py
```
服務器將在 `0.0.0.0:50500` 上運行。

### 測試情緒辨識
- 打開您的瀏覽器，並訪問 `http://localhost:50500`。
- 上傳一個音訊檔案，然後點擊「預測」按鈕。

## 文件結構
- `main.py`：主程式檔案，包含Flask應用和所有API端點。
- `TIM.py`：TIMNet模型的實現。
- `config.py`：存放所有配置參數。

## API
### `POST /predict`
接受一個音訊檔案，返回預測的情緒。

### `GET /input`
返回音訊上傳頁面。

## License
本專案使用Apache-2.0許可證，詳見[LICENSE](LICENSE)檔案。
```
