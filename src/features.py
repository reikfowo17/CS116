import pandas as pd
import numpy as np
import gc
from config import GROUP_COLS, KEY_FEATURES, EXTRA_FEATURES, LAG_STEPS, ROLLING_WINDOWS


def compute_target_stats(df):
    """Compute target encoding stats from training data only."""
    stats = {
        'code': df.groupby('code')['y_target'].mean().to_dict(),
        'sub_category': df.groupby('sub_category')['y_target'].mean().to_dict(),
        'sub_code': df.groupby('sub_code')['y_target'].mean().to_dict(),
        'global_mean': float(df['y_target'].mean()),
    }
    return stats


def compute_freq_encoding(df):
    """Compute frequency encoding from data (rule-safe, no target used)."""
    freq = {}
    for c in ['code', 'sub_code', 'sub_category']:
        if c in df.columns:
            freq[c] = df[c].value_counts(normalize=True).to_dict()
    return freq


def build_features(data, target_stats, freq_stats):
    """Build all features. Handles concat train+test data (y_target can be NaN)."""
    df = data.copy()

    # ── 1. Feature Interactions ──
    if 'feature_al' in df.columns and 'feature_am' in df.columns:
        df['d_al_am'] = df['feature_al'] - df['feature_am']
        df['r_al_am'] = df['feature_al'] / (df['feature_am'] + 1e-7)
    if 'feature_cg' in df.columns and 'feature_by' in df.columns:
        df['d_cg_by'] = df['feature_cg'] - df['feature_by']
    if 'feature_s' in df.columns:
        for f in ['feature_al', 'feature_am', 'feature_cg']:
            if f in df.columns:
                df[f's_{f.split("_")[1]}_prod'] = df['feature_s'] * df[f]

    # ── 2. Target Encoding (from train stats) ──
    for c in ['code', 'sub_category', 'sub_code']:
        if c in df.columns and c in target_stats:
            df[c + '_enc'] = df[c].map(target_stats[c]).fillna(target_stats['global_mean']).astype(np.float32)

    # ── 3. Frequency Encoding (rule-safe, no target) ──
    for c in ['code', 'sub_code', 'sub_category']:
        if c in df.columns and c in freq_stats:
            df[c + '_freq'] = df[c].map(freq_stats[c]).fillna(0).astype(np.float32)

    # ── 4. Cross-sectional normalization ──
    cs_cols = [c for c in ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'd_al_am'] if c in df.columns]
    for col in cs_cols:
        g = df.groupby('ts_index')[col]
        g_mean = g.transform('mean')
        g_std = g.transform('std') + 1e-7
        df[col + '_cs'] = ((df[col] - g_mean) / g_std).astype(np.float32)

    # ── 5. Time features ──
    for p in [2, 3, 5, 7, 12, 24, 30]:
        df[f'ts_mod_{p}'] = (df['ts_index'] % p).astype(np.int8)

    # ── 6. Lifecycle ──
    df = df.sort_values(GROUP_COLS + ['ts_index'])
    df['obs_idx'] = df.groupby(GROUP_COLS).cumcount().astype(np.int32)
    first_t = df.groupby(GROUP_COLS)['ts_index'].transform('min')
    df['time_since_start'] = (df['ts_index'] - first_t).astype(np.int32)
    del first_t

    # ── 7. Lags, Rolling, EWM, Diff, Rank ──
    target_cols = [c for c in KEY_FEATURES + EXTRA_FEATURES if c in df.columns]
    grouped = df.groupby(GROUP_COLS, sort=False)
    new = {}

    for col in target_cols:
        # Lags
        for lag in LAG_STEPS:
            new[f'{col}_lag{lag}'] = grouped[col].shift(lag).astype(np.float32)

        # Rolling (shift 1 first to avoid look-ahead)
        for w in ROLLING_WINDOWS:
            shifted = grouped[col].shift(1)
            new[f'{col}_rmean_{w}'] = shifted.rolling(w, min_periods=1).mean().astype(np.float32)
            new[f'{col}_rstd_{w}'] = shifted.rolling(w, min_periods=1).std().astype(np.float32)

        # EWM — FAST version: shift(1) + ewm, no slow lambda
        new[f'{col}_ewm10'] = grouped[col].shift(1).ewm(
            span=10, min_periods=1
        ).mean().values.astype(np.float32)

        # Diff
        new[f'{col}_diff1'] = grouped[col].diff(1).astype(np.float32)

        # Rank (cross-sectional per timestep)
        new[f'{col}_rank'] = df.groupby('ts_index')[col].rank(pct=True).astype(np.float32)

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # ── 8. Momentum ──
    for col in KEY_FEATURES:
        l1 = f'{col}_lag1'
        l5 = f'{col}_lag5'
        rm5 = f'{col}_rmean_5'
        if l1 in df.columns and l5 in df.columns:
            df[f'{col}_mom15'] = (df[l1] - df[l5]).astype(np.float32)
        if l1 in df.columns and rm5 in df.columns:
            df[f'{col}_dev5'] = (df[l1] - df[rm5]).astype(np.float32)

    # ── 9. Fill NaN/Inf (preserve y_target & weight NaN for concat split) ──
    preserved_target = df['y_target'].copy() if 'y_target' in df.columns else None
    preserved_weight = df['weight'].copy() if 'weight' in df.columns else None

    df = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    if preserved_target is not None:
        df['y_target'] = preserved_target
    if preserved_weight is not None:
        df['weight'] = preserved_weight

    gc.collect()
    return df


def get_feature_columns(df):
    exclude = {'id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'weight', 'y_target'}
    return [c for c in df.columns if c not in exclude]
