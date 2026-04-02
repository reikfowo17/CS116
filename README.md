# 🏆 Kaggle: Hedge Fund Time Series Forecasting

## Team Members
- [Le Gia Bao]
- [Hoang Duc Bao]

## Competition
[Hedge Fund - Time Series Forecasting](https://kaggle.com/competitions/ts-forecasting)

### Project Structure
```
├── src/
│   ├── config.py         # Paths, seeds, hyperparameters
│   ├── data_loader.py    # Load & preprocess parquet data
│   ├── features.py       # Feature engineering pipeline
│   ├── models.py         # Training, CV, stacking, prediction
│   └── evaluation.py     # Weighted RMSE scoring
├── docs/
│   └── experiment_log.md # Experiment tracking
├── requirements.txt
└── README.md
```

## Hướng Dẫn Chạy Trên Kaggle

### Bước 1: Clone Code
Tạo Notebook mới trên Kaggle. Thêm GitHub Token vào **Add-ons → Secrets** (label: `GITHUB_TOKEN`).
```python
# Cell 1: Clone repo
from kaggle_secrets import UserSecretsClient
import os, sys

secrets = UserSecretsClient()
github_token = secrets.get_secret("GITHUB_TOKEN")

os.system("rm -rf /kaggle/working/repo")
os.system(f"git clone https://{github_token}@github.com/reikfowo17/CS116.git /kaggle/working/repo")

sys.path.append('/kaggle/working/repo/src')
```

### Bước 2: Train & Predict
Đảm bảo đã thêm dataset chứa `train.parquet` và `test.parquet` vào notebook.
```python
# Cell 2: Train & Predict
from models import train_and_predict_all_horizons, create_submission

sub, scores = train_and_predict_all_horizons()
```

### Bước 3: Xuất Submission
```python
# Cell 3: Tạo file submission.csv
create_submission(sub, "submission.csv")
```

## Local Setup
```bash
pip install -r requirements.txt
```
