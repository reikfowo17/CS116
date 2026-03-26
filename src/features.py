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
        'sub_code_q10': df.groupby('sub_code')['y_target'].quantile(0.10).to_dict(),
        'sub_code_q90': df.groupby('sub_code')['y_target'].quantile(0.90).to_dict(),
        'sub_code_std': df.groupby('sub_code')['y_target'].std().fillna(1.0).to_dict(),
        'sub_cat_std': df.groupby('sub_category')['y_target'].std().fillna(1.0).to_dict(),
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
    gm = target_stats.get('global_mean', 0.0)
    raw_cols = [c for c in df.columns if c.startswith('feature_')]

    # ── 1. Row-wise meta-features (numpy for speed) ──
    if raw_cols:
        X = df[raw_cols].fillna(0.0).values
        df['feat_mean']     = np.mean(X, axis=1).astype(np.float32)
        df['feat_std']      = np.std(X, axis=1).astype(np.float32)
        df['feat_range']    = (np.max(X, axis=1) - np.min(X, axis=1)).astype(np.float32)
        df['feat_pos_frac'] = (X > 0).mean(axis=1).astype(np.float32)
        df['feat_l2']       = np.sqrt((X ** 2).sum(axis=1)).astype(np.float32)
        del X

    # ── 2. Feature Interactions (expanded) ──
    E = lambda f: f in df.columns
    if E('feature_al') and E('feature_am'):
        df['d_al_am'] = df['feature_al'] - df['feature_am']
        df['r_al_am'] = df['feature_al'] / (df['feature_am'].abs() + 1e-7)
        df['p_al_am'] = df['feature_al'] * df['feature_am']
    if E('feature_cg') and E('feature_by'):
        df['d_cg_by'] = df['feature_cg'] - df['feature_by']
        df['r_cg_by'] = df['feature_cg'] / (df['feature_by'].abs() + 1e-7)
        df['mean_cg_by'] = (df['feature_cg'] + df['feature_by']) / 2.0
    if E('feature_al') and E('feature_bp'):
        df['d_al_bp'] = df['feature_al'] - df['feature_bp']
        df['mean_al_bp'] = (df['feature_al'] + df['feature_bp']) / 2.0
        df['r_al_bp'] = df['feature_al'] / (df['feature_bp'].abs() + 1e-7)
    if E('feature_s') and E('feature_t'):
        df['d_s_t'] = df['feature_s'] - df['feature_t']
    if E('feature_s'):
        for f in ['feature_al', 'feature_am', 'feature_cg']:
            if E(f):
                df[f's_{f.split("_")[1]}_prod'] = df['feature_s'] * df[f]
    if E('feature_am') and E('feature_bz'):
        df['p_am_bz'] = df['feature_am'] * df['feature_bz']
    if E('feature_al') and E('feature_cg'):
        df['al_x_cg'] = df['feature_al'] * df['feature_cg']
    if E('feature_a') and E('feature_b'):
        df['d_a_b'] = df['feature_a'] - df['feature_b']
    if E('feature_c') and E('feature_d'):
        df['d_c_d'] = df['feature_c'] - df['feature_d']
    if E('feature_e') and E('feature_f'):
        df['d_e_f'] = df['feature_e'] - df['feature_f']
    if all(E(c) for c in ['feature_al', 'feature_bp', 'feature_am', 'feature_bq']):
        df['wap'] = (df['feature_al'] * df['feature_bq'] + df['feature_bp'] * df['feature_am']) / (df['feature_am'] + df['feature_bq'] + 1e-7)

    # ── 3. Target Encoding (from train stats) ──
    for c in ['code', 'sub_category', 'sub_code']:
        if c in target_stats:
            df[c + '_enc'] = df[c].map(target_stats[c]).fillna(gm).astype(np.float32)
    df['sc_q10']    = df['sub_code'].map(target_stats.get('sub_code_q10', {})).fillna(gm).astype(np.float32)
    df['sc_q90']    = df['sub_code'].map(target_stats.get('sub_code_q90', {})).fillna(gm).astype(np.float32)
    df['sc_qrange'] = (df['sc_q90'] - df['sc_q10']).astype(np.float32)
    df['sc_std']    = df['sub_code'].map(target_stats.get('sub_code_std', {})).fillna(1.0).astype(np.float32)
    df['scat_std']  = df['sub_category'].map(target_stats.get('sub_cat_std', {})).fillna(1.0).astype(np.float32)
    if 'sub_code_enc' in df.columns:
        df['code_snr']  = (df['sub_code_enc'].abs() / (df['sc_std'] + 1e-7)).astype(np.float32)

    # ── 4. Frequency Encoding (rule-safe, no target) ──
    for c in ['code', 'sub_code', 'sub_category']:
        if c in freq_stats:
            df[c + '_freq'] = df[c].map(freq_stats[c]).fillna(0).astype(np.float32)
    if 'sub_code_freq' in df.columns:
        df['sc_log_freq'] = np.log1p(df['sub_code_freq']).astype(np.float32)

    # ── 5. Cross-sectional normalization + rank ──
    cs_cols = [c for c in ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'd_al_am', 'd_cg_by', 'feat_mean'] if E(c)]
    for col in cs_cols:
        g = df.groupby('ts_index')[col]
        df[col + '_cs'] = ((df[col] - g.transform('mean')) / (g.transform('std') + 1e-7)).astype(np.float32)
        df[col + '_rank'] = g.rank(pct=True).astype(np.float32)
    if E('feature_s'):
        df['feature_s_rank'] = df.groupby('ts_index')['feature_s'].rank(pct=True).astype(np.float32)
    for f in ['feature_al', 'feature_am', 'd_al_am']:
        if E(f):
            g2 = df.groupby(['ts_index', 'sub_category'])[f]
            df[f + '_gcs'] = ((df[f] - g2.transform('mean')) / (g2.transform('std') + 1e-7)).astype(np.float32)

    # ── 6. Time features ──
    ts = df['ts_index']
    for p in [2, 3, 5, 7, 12, 24, 30]:
        df[f'ts_mod_{p}'] = (ts % p).astype(np.int8)
    df['t_sin']  = np.sin(2 * np.pi * ts / 100).astype(np.float32)
    df['t_cos']  = np.cos(2 * np.pi * ts / 100).astype(np.float32)
    df['t_sin2'] = np.sin(2 * np.pi * ts / 52).astype(np.float32)
    df['t_cos2'] = np.cos(2 * np.pi * ts / 52).astype(np.float32)
    df['ts_norm'] = (ts / 4000.0).astype(np.float32)
    if 'horizon' in df.columns:
        df['horizon_log'] = np.log1p(df['horizon']).astype(np.float32)

    # ── 7. Lifecycle ──
    df = df.sort_values(GROUP_COLS + ['ts_index'])
    df['obs_idx'] = df.groupby(GROUP_COLS).cumcount().astype(np.int32)
    first_t = df.groupby(GROUP_COLS)['ts_index'].transform('min')
    df['time_since_start'] = (df['ts_index'] - first_t).astype(np.int32)
    del first_t

    # ── 8. Lags, Rolling, EWM, Diff, Rank ──
    key_present = [f for f in KEY_FEATURES if E(f)]
    extra_present = [f for f in EXTRA_FEATURES if E(f)]
    inter_feats = [f for f in ['d_al_am', 'd_cg_by', 'd_s_t', 'mean_al_bp', 'mean_cg_by', 'd_al_bp', 'wap'] if E(f)]
    
    grouped = df.groupby(GROUP_COLS, sort=False)
    new = {}

    for col in key_present:
        for lag in LAG_STEPS:
            new[f'{col}_lag{lag}'] = grouped[col].shift(lag).astype(np.float32)
        for w in ROLLING_WINDOWS:
            shifted = grouped[col].shift(1)
            new[f'{col}_rmean_{w}'] = shifted.rolling(w, min_periods=1).mean().values.astype(np.float32)
            new[f'{col}_rstd_{w}']  = shifted.rolling(w, min_periods=1).std().values.astype(np.float32)
        
        # SẠCH VÀ CHUẨN: shift(1) cho hàm EWM
        new[f'{col}_ewm10'] = grouped[col].shift(1).ewm(span=10, min_periods=1).mean().values.astype(np.float32)
        new[f'{col}_diff1'] = grouped[col].diff(1).astype(np.float32)

    for col in extra_present:
        for lag in [1, 3, 5]:
            new[f'{col}_lag{lag}'] = grouped[col].shift(lag).astype(np.float32)
        new[f'{col}_diff1'] = grouped[col].diff(1).astype(np.float32)
        
        shifted = grouped[col].shift(1)
        new[f'{col}_rmean_5'] = shifted.rolling(5, min_periods=1).mean().values.astype(np.float32)
        new[f'{col}_ewm10']   = grouped[col].shift(1).ewm(span=10, min_periods=1).mean().values.astype(np.float32)

    for col in inter_feats:
        for lag in [1, 3]:
            new[f'{col}_lag{lag}'] = grouped[col].shift(lag).astype(np.float32)
        new[f'{col}_diff1'] = grouped[col].diff(1).astype(np.float32)

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # ── 9. Momentum ──
    for col in key_present:
        l1, l5, rm5 = f'{col}_lag1', f'{col}_lag5', f'{col}_rmean_5'
        if E(l1) and E(l5):
            df[f'{col}_mom15'] = (df[l1] - df[l5]).astype(np.float32)
        if E(l1) and E(rm5):
            df[f'{col}_dev5']  = (df[l1] - df[rm5]).astype(np.float32)

    # ── 10. Fill NaN/Inf ──
    preserved_y = df['y_target'].copy() if 'y_target' in df.columns else None
    preserved_w = df['weight'].copy() if 'weight' in df.columns else None
    
    df = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    if preserved_y is not None: df['y_target'] = preserved_y
    if preserved_w is not None: df['weight'] = preserved_w

    gc.collect()
    return df


def get_feature_columns(df):
    exclude = {'id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'weight', 'y_target'}
    return [c for c in df.columns if c not in exclude]
