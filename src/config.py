import os
import subprocess

# ── Environment Detection ──
IS_KAGGLE = os.path.exists('/kaggle/input')

# GPU Detection
def _has_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

HAS_GPU = _has_gpu()

# ── Data Paths ──
if IS_KAGGLE:
    TRAIN_PATH = "/kaggle/input/competitions/ts-forecasting/train.parquet"
    TEST_PATH  = "/kaggle/input/competitions/ts-forecasting/test.parquet"
    OUTPUT_DIR = "/kaggle/working"
else:
    TRAIN_PATH = "data/train.parquet"
    TEST_PATH  = "data/test.parquet"
    OUTPUT_DIR = "submissions"

# ── Constants ──
SEED = 42
HORIZONS = [1, 3, 10, 25]
TARGET = "y_target"
WEIGHT = "weight"

# ── Column Groups ──
CAT_COLS   = ["code", "sub_code", "sub_category"]
META_COLS  = ["id", "code", "sub_code", "sub_category", "horizon", "ts_index"]
GROUP_COLS = ["code", "sub_code", "sub_category", "horizon"]

# ── Feature Engineering Config ──
# Key features identified from competitor analysis
KEY_FEATURES = ["feature_al", "feature_am", "feature_cg", "feature_by"]
EXTRA_FEATURES = ["feature_s"]  # optional, check if exists
LAG_STEPS = [1, 3, 5, 10, 25]
ROLLING_WINDOWS = [5, 10, 20]

# ── Validation Split ──
VAL_THRESHOLD = 3500  # ts_index <= 3500 = train, > 3500 = val

# ── Multi-Seed Ensemble ──
SEEDS = [42, 2024, 12345, 99, 420, 777, 1337, 2025, 7, 11,
         13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

# ── Clipping ──
CLIP_Q_LOW = 0.005
CLIP_Q_HIGH = 0.995

# ── LightGBM Base Params ──
LGBM_BASE_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "boosting_type":    "gbdt",
    "learning_rate":    0.015,
    "num_leaves":       90,
    "max_depth":        -1,
    "min_child_samples": 200,
    "feature_fraction": 0.65,
    "bagging_fraction": 0.75,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        10.0,
    "verbosity":        -1,
    "n_jobs":           -1,
}

# ── XGBoost Base Params ──
XGB_BASE_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "learning_rate":    0.015,
    "max_depth":        6,
    "min_child_weight": 200,
    "subsample":        0.75,
    "colsample_bytree": 0.65,
    "reg_alpha":        0.1,
    "reg_lambda":       10.0,
    "tree_method":      "hist",
    "device":            "cuda" if HAS_GPU else "cpu",
    "verbosity":        0,
    "n_jobs":           -1,
}

# ── XGBoost Seeds ──
XGB_SEEDS = [42, 2024, 12345, 99, 420, 777, 1337, 2025, 7, 11]

# ── Blend Weights ──
BLEND_W_LGB = 0.85
BLEND_W_XGB = 0.15
