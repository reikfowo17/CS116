# 🏆 Kaggle: Hedge Fund Time Series Forecasting

## Team Members
- [Le Gia Bao]
- [Hoang Duc Bao]

## Competition
[Hedge Fund - Time Series Forecasting](https://kaggle.com/competitions/ts-forecasting)

## Project Structure
```
├── notebooks/          # Kaggle notebooks
├── src/                # Reusable Python modules
├── submissions/        # CSV submission files
├── docs/               # Documentation & logs
├── requirements.txt    # Dependencies
└── .gitignore
```

## 🚀 Hướng Dẫn Chạy & Kiểm Thử Trên Kaggle

### Bước 1: Kéo Code Từ GitHub Về
Tạo một Notebook mới trên Kaggle. Trong cell đầu tiên, clone repo này về:
```python
# Cell 1: Clone repo
!rm -rf /kaggle/working/repo
!git clone https://github.com/reikfowo17/CS116.git /kaggle/working/repo

import sys
sys.path.append('/kaggle/working/repo/src')
```

### Bước 2: Import Modules Gắn Với Dữ Liệu
Hãy đảm bảo phần Data bên góc phải Kaggle Notebook của bạn đã thêm tập dataset chứa `train.parquet` và `test.parquet`.
```python
# Cell 2: Import các hàm cấu trúc chính
from config import *
from data_loader import load_data
from features import engineer_features
from models import train_per_horizon, predict_per_horizon, create_submission
from evaluation import weighted_rmse_score
```

### Bước 3: Load và Feature Engineering
```python
# Load data
train_df, test_df = load_data()

# Tiến hành sinh Features (Lag, Rolling, Diff...) 
train_df, test_df = engineer_features(
    train_df, test_df,
    lag_features=TOP_LAG_FEATURES, lags=LAG_STEPS,
    rolling_features=TOP_LAG_FEATURES[:3], windows=ROLLING_WINDOWS,
    diff_features=TOP_LAG_FEATURES[:3],
)
```

### Bước 4: Chạy Huấn Luyện & Đánh Giá
Tách rời 4 mô hình để huấn luyện riêng rẽ cho từng **Horizon (1, 3, 10, 25)**
```python
# Tách các feature sử dụng cho đầu vào model
features = [c for c in train_df.columns if c not in ["id", "y_target", "weight"]]
print(f"Tổng số features dùng để train: {len(features)}")

# Huấn luyện mô hình và lấy điểm nội suy
models, scores = train_per_horizon(
    train_df, features, split_ts_index=3000
)
# Model sẽ in ra score (0 đến 1) của từng horizon.
```

### Bước 5: Dự Đoán Kết Quả & Xuất File Submission
```python
# Tiến hành Predict trên tập test:
from utils import check_submission

submission = predict_per_horizon(models, test_df, features)
check_submission(submission, test_df)

# Tạo file "submission.csv" để Kaggle lấy dữ liệu chấm điểm
create_submission(submission, "submission.csv")
```

---

## 💻 Local Setup
```bash
pip install -r requirements.txt
```
