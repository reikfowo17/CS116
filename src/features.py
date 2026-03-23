import pandas as pd
import numpy as np
import gc
from config import GROUP_COLS, KEY_FEATURES, EXTRA_FEATURES, LAG_STEPS, ROLLING_WINDOWS


def compute_target_encoding_stats(df):
    stats = {
        'sub_category': df.groupby('sub_category')['y_target'].mean().to_dict(),
        'sub_code': df.groupby('sub_code')['y_target'].mean().to_dict(),
        'global_mean': float(df['y_target'].mean()),
    }
    return stats


def build_features(data, target_stats):
    df = data.copy()
    group_cols = GROUP_COLS

    # ── 1. Feature Interactions ──
    if 'feature_al' in df.columns and 'feature_am' in df.columns:
        df['d_al_am'] = df['feature_al'] - df['feature_am']
        df['r_al_am'] = df['feature_al'] / (df['feature_am'] + 1e-7)
    if 'feature_cg' in df.columns and 'feature_by' in df.columns:
        df['d_cg_by'] = df['feature_cg'] - df['feature_by']

    if 'feature_s' in df.columns:
        if 'feature_al' in df.columns:
            df['s_al_prod'] = df['feature_s'] * df['feature_al']
        if 'feature_am' in df.columns:
            df['s_am_prod'] = df['feature_s'] * df['feature_am']
        if 'feature_cg' in df.columns:
            df['s_cg_prod'] = df['feature_s'] * df['feature_cg']

    # ── 2. Target Encoding ──
    for c in ['sub_category', 'sub_code']:
        if c in df.columns:
            df[c + '_enc'] = df[c].map(target_stats[c]).fillna(target_stats['global_mean']).astype(np.float32)

    # ── 3. Cross-sectional normalization ──
    cs_cols = ['feature_al', 'feature_am', 'feature_cg', 'feature_by']
    if 'd_al_am' in df.columns:
        cs_cols.append('d_al_am')
    for col in cs_cols:
        if col in df.columns:
            g = df.groupby('ts_index')[col]
            g_mean = g.transform('mean')
            g_std = g.transform('std') + 1e-7
            df[col + '_cs'] = ((df[col] - g_mean) / g_std).astype(np.float32)
            del g_mean, g_std

    # ── 4. Time phase features ──
    for p in [2, 3, 4, 5, 7, 12, 14, 24, 28, 30]:
        df[f'ts_mod_{p}'] = (df['ts_index'] % p).astype(np.int8)

    # ── 5. Group lifecycle features ──
    df = df.sort_values(group_cols + ['ts_index'])
    df['obs_idx_in_group'] = df.groupby(group_cols).cumcount().astype(np.int32)
    first_time = df.groupby(group_cols)['ts_index'].transform('min')
    df['time_since_group_start'] = (df['ts_index'] - first_time).astype(np.int32)
    del first_time

    # ── 6. Time-series engine (lags, rolling, ewm, diff, rank) ──
    target_cols = [c for c in KEY_FEATURES if c in df.columns]
    for ef in EXTRA_FEATURES:
        if ef in df.columns:
            target_cols.append(ef)

    grouped = df.groupby(group_cols, sort=False)

    for col in target_cols:
        # Lags
        for lag in LAG_STEPS:
            df[f'{col}_lag{lag}'] = grouped[col].shift(lag).astype(np.float32)

        # Rolling mean + std + min + max (shift(1) before rolling)
        for w in ROLLING_WINDOWS:
            shifted = grouped[col].shift(1)
            df[f'{col}_roll_mean_{w}'] = shifted.rolling(w, min_periods=1).mean().astype(np.float32)
            df[f'{col}_roll_std_{w}'] = shifted.rolling(w, min_periods=1).std().astype(np.float32)
            df[f'{col}_roll_min_{w}'] = shifted.rolling(w, min_periods=1).min().astype(np.float32)
            df[f'{col}_roll_max_{w}'] = shifted.rolling(w, min_periods=1).max().astype(np.float32)

        # EWM
        df[f'{col}_ewm_10'] = grouped[col].transform(
            lambda x: x.ewm(span=10, min_periods=1).mean()
        ).astype(np.float32)

        # Diff
        df[f'{col}_diff1'] = grouped[col].diff(1).astype(np.float32)

        # Rank (cross-sectional)
        df[f'{col}_rank'] = df.groupby('ts_index')[col].rank(pct=True).astype(np.float32)

    # ── 7. Momentum features ──
    for col in KEY_FEATURES:
        lag1_col = f'{col}_lag1'
        lag5_col = f'{col}_lag5'
        roll_mean_5 = f'{col}_roll_mean_5'
        if lag1_col in df.columns and lag5_col in df.columns:
            df[f'{col}_mom_1_5'] = (df[lag1_col] - df[lag5_col]).astype(np.float32)
        if lag1_col in df.columns and roll_mean_5 in df.columns:
            df[f'{col}_dev_from_roll5'] = (df[lag1_col] - df[roll_mean_5]).astype(np.float32)

    gc.collect()

    # ── 8. Fill NaN/Inf ──
    df = df.fillna(0.0)
    df = df.replace([np.inf, -np.inf], 0.0)

    return df


def get_feature_columns(df):
    exclude = {'id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'weight', 'y_target'}
    return [c for c in df.columns if c not in exclude]
