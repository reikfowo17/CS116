import os

# ── Environment Detection ──
IS_KAGGLE = os.path.exists('/kaggle/input')

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

# ── Cross-Validation ──
N_CV_SPLITS = 5

# ── Seeds per CV fold ──
CV_LGB_SEEDS = [42, 2024, 12345, 99, 420]
CV_CAT_SEEDS = [42, 2024, 12345]

# ── Seeds for final retrain-on-all ──
FINAL_LGB_SEEDS = [42, 2024, 12345, 99, 420, 777, 1337, 2025, 7, 11,
                   13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
FINAL_CAT_SEEDS = [42, 2024, 12345, 99, 420, 777, 1337, 2025, 7, 11]


# ── LightGBM Base Params ──
LGBM_BASE_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "boosting_type":    "gbdt",
    "learning_rate":    0.015,
    "num_leaves":       127,
    "max_depth":        -1,
    "min_child_samples": 200,
    "feature_fraction": 0.60,
    "bagging_fraction": 0.75,
    "bagging_freq":     5,
    "lambda_l1":        2.0,
    "lambda_l2":        10.0,
    "extra_trees":      True,
    "path_smooth":      1.0,
    "verbosity":        -1,
    "n_jobs":           -1,
}

# ── CatBoost Base Params ──
CATBOOST_PARAMS = {
    "iterations":       3000,
    "learning_rate":    0.03,
    "depth":            7,
    "l2_leaf_reg":      3.0,
    "verbose":          0,
    "early_stopping_rounds": 200,
}

# ── CatBoost Seeds ──
CAT_SEEDS = [42, 2024, 12345, 99, 420, 777, 1337, 2025, 7, 11]

