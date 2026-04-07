# Hedge Fund Time Series Forecasting

## Competition
[Hedge Fund - Time Series Forecasting](https://kaggle.com/competitions/ts-forecasting)

## Team Members
- Le Gia Bao
- Hoang Duc Bao

## Project Structure
```
CS116/
├── src/
│   ├── config.py         # Paths, seeds, hyperparameters
│   ├── data_loader.py    # Load & preprocess parquet data
│   ├── features.py       # Feature engineering pipeline
│   ├── models.py         # Training, cross-validation, prediction
│   └── evaluation.py     # Weighted RMSE scoring
├── docs/
│   └── experiment_log.md # Experiment tracking
├── requirements.txt
└── README.md
```

## How to Run on Kaggle

### Step 1: Clone Code
Create a new Notebook. Add GitHub Token to **Add-ons → Secrets** (label: `GITHUB_TOKEN`).

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

### Step 2: Train & Predict
Make sure the dataset containing `train.parquet` and `test.parquet` is added to the notebook.

```python
# Cell 2: Train & Predict
from models import train_and_predict_all_horizons, create_submission

sub, scores = train_and_predict_all_horizons()
```

### Step 3: Export Submission

```python
# Cell 3: Create submission.csv
create_submission(sub, "submission.csv")
```

## Local Setup

```bash
pip install -r requirements.txt
```

## Development Log

See `docs/experiment_log.md` for full experiment history and scores.
