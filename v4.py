import os, gc, time
import numpy as np
import pandas as pd
import lightgbm as lgb

# ══════════════════════════════════════════════════════════════════
# CONFIG — V4 ENHANCED
# ══════════════════════════════════════════════════════════════════
IS_KAGGLE = os.path.exists("/kaggle/input")
if IS_KAGGLE:
    TRAIN_PATH = "/kaggle/input/competitions/ts-forecasting/train.parquet"
    TEST_PATH  = "/kaggle/input/competitions/ts-forecasting/test.parquet"
    OUTPUT_DIR = "/kaggle/working"
else:
    TRAIN_PATH = "train.parquet"
    TEST_PATH  = "test.parquet"
    OUTPUT_DIR = "."

HORIZONS       = [1, 3, 10, 25]
TARGET         = "y_target"
WEIGHT         = "weight"
VAL_THRESHOLD  = 3500
GROUP_COLS     = ["code", "sub_code", "sub_category", "horizon"]

# Optimal feature shifts per horizon (from eda-v2, eda-hedge-fund1 notebooks)
# feature shifted by -h means "value from h steps ago" = causal pseudo-target
OPTIMAL_SHIFTS = {1: -1, 3: -3, 10: -10, 25: -25}

KEY_FEATURES   = ["feature_al", "feature_am", "feature_cg", "feature_by", "mean_al_bp"]
EXTRA_FEATURES = ["feature_s"]
LAG_STEPS      = [1, 3, 5, 10, 25]
ROLLING_WINDOWS = [5, 10, 20]

N_SEEDS = 7
SEEDS   = [42, 2024, 12345, 99, 420, 7, 2025]

SCALE_GRID = [0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]

TOP_FEATURES = 100

LGB_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "boosting_type":     "gbdt",
    "learning_rate":     0.015,
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 200,
    "feature_fraction":  0.60,
    "bagging_fraction":  0.75,
    "bagging_freq":      5,
    "lambda_l1":         2.0,
    "lambda_l2":         10.0,
    "extra_trees":       True,
    "path_smooth":       1.0,
    "verbosity":         -1,
    "n_jobs":            -1,
}


# ══════════════════════════════════════════════════════════════════
# METRIC
# ══════════════════════════════════════════════════════════════════
def weighted_rmse_score(y_true, y_pred, w):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w      = np.asarray(w, dtype=np.float64)
    denom  = np.sum(w * y_true ** 2)
    if denom == 0:
        return 0.0
    ratio = np.sum(w * (y_true - y_pred) ** 2) / denom
    return float(np.sqrt(max(0.0, 1.0 - min(ratio, 1.0))))


# ══════════════════════════════════════════════════════════════════
# TARGET STATS
# ══════════════════════════════════════════════════════════════════
def compute_target_stats(df):
    return {
        'code':          df.groupby('code')[TARGET].mean().to_dict(),
        'sub_category':  df.groupby('sub_category')[TARGET].mean().to_dict(),
        'sub_code':      df.groupby('sub_code')[TARGET].mean().to_dict(),
        'global_mean':   float(df[TARGET].mean()),
        'sub_code_q10':  df.groupby('sub_code')[TARGET].quantile(0.10).to_dict(),
        'sub_code_q90':  df.groupby('sub_code')[TARGET].quantile(0.90).to_dict(),
        'sub_code_std':  df.groupby('sub_code')[TARGET].std().fillna(1.0).to_dict(),
        'sub_cat_std':   df.groupby('sub_category')[TARGET].std().fillna(1.0).to_dict(),
    }


def compute_freq_encoding(df):
    freq = {}
    for c in ['code', 'sub_code', 'sub_category']:
        if c in df.columns:
            freq[c] = df[c].value_counts(normalize=True).to_dict()
    return freq


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — V4 ENHANCED (~185 features, +pseudo-targets)
# ══════════════════════════════════════════════════════════════════
def build_features(data, target_stats, freq_stats):
    df = data.copy()
    gm = target_stats.get('global_mean', 0.0)
    raw_cols = [c for c in df.columns if c.startswith('feature_')]
    E = lambda f: f in df.columns

    # 1. Row-wise meta features
    if raw_cols:
        X = df[raw_cols].fillna(0.0).values
        df['feat_mean']     = np.mean(X, axis=1).astype(np.float32)
        df['feat_std']      = np.std(X, axis=1).astype(np.float32)
        df['feat_range']    = (np.max(X, axis=1) - np.min(X, axis=1)).astype(np.float32)
        df['feat_pos_frac'] = (X > 0).mean(axis=1).astype(np.float32)
        df['feat_l2']       = np.sqrt((X ** 2).sum(axis=1)).astype(np.float32)
        del X

    # 2. Feature interactions
    if E('feature_al') and E('feature_am'):
        df['d_al_am'] = df['feature_al'] - df['feature_am']
        df['r_al_am'] = df['feature_al'] / (df['feature_am'].abs() + 1e-7)
        df['p_al_am'] = df['feature_al'] * df['feature_am']
    if E('feature_cg') and E('feature_by'):
        df['d_cg_by']    = df['feature_cg'] - df['feature_by']
        df['r_cg_by']    = df['feature_cg'] / (df['feature_by'].abs() + 1e-7)
        df['mean_cg_by'] = (df['feature_cg'] + df['feature_by']) / 2.0
    if E('feature_al') and E('feature_bp'):
        df['d_al_bp']    = df['feature_al'] - df['feature_bp']
        df['mean_al_bp'] = (df['feature_al'] + df['feature_bp']) / 2.0
        df['r_al_bp']    = df['feature_al'] / (df['feature_bp'].abs() + 1e-7)
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
        df['wap'] = (df['feature_al'] * df['feature_bq'] + df['feature_bp'] * df['feature_am']) / \
                    (df['feature_am'] + df['feature_bq'] + 1e-7)

    # 3. Target encoding (from train stats ONLY)
    for c in ['code', 'sub_category', 'sub_code']:
        if c in target_stats:
            df[c + '_enc'] = df[c].map(target_stats[c]).fillna(gm).astype(np.float32)
    df['sc_q10']    = df['sub_code'].map(target_stats.get('sub_code_q10', {})).fillna(gm).astype(np.float32)
    df['sc_q90']    = df['sub_code'].map(target_stats.get('sub_code_q90', {})).fillna(gm).astype(np.float32)
    df['sc_qrange'] = (df['sc_q90'] - df['sc_q10']).astype(np.float32)
    df['sc_std']    = df['sub_code'].map(target_stats.get('sub_code_std', {})).fillna(1.0).astype(np.float32)
    df['scat_std']  = df['sub_category'].map(target_stats.get('sub_cat_std', {})).fillna(1.0).astype(np.float32)
    if 'sub_code_enc' in df.columns:
        df['code_snr'] = (df['sub_code_enc'].abs() / (df['sc_std'] + 1e-7)).astype(np.float32)

    # 4. Frequency encoding
    for c in ['code', 'sub_code', 'sub_category']:
        if c in freq_stats:
            df[c + '_freq'] = df[c].map(freq_stats[c]).fillna(0).astype(np.float32)
    if 'sub_code_freq' in df.columns:
        df['sc_log_freq'] = np.log1p(df['sub_code_freq']).astype(np.float32)

    # 5. Cross-sectional normalization + rank
    cs_cols = [c for c in ['feature_al', 'feature_am', 'feature_cg', 'feature_by',
                           'mean_al_bp', 'd_al_am', 'd_cg_by', 'feat_mean'] if E(c)]
    for col in cs_cols:
        g = df.groupby('ts_index')[col]
        df[col + '_cs']   = ((df[col] - g.transform('mean')) / (g.transform('std') + 1e-7)).astype(np.float32)
        df[col + '_rank'] = g.rank(pct=True).astype(np.float32)
    if E('feature_s'):
        df['feature_s_rank'] = df.groupby('ts_index')['feature_s'].rank(pct=True).astype(np.float32)
    for f in ['feature_al', 'feature_am', 'd_al_am', 'mean_al_bp']:
        if E(f):
            g2 = df.groupby(['ts_index', 'sub_category'])[f]
            df[f + '_gcs'] = ((df[f] - g2.transform('mean')) / (g2.transform('std') + 1e-7)).astype(np.float32)

    # 6. Time features
    ts = df['ts_index']
    for p in [2, 3, 5, 7, 12, 24, 30]:
        df[f'ts_mod_{p}'] = (ts % p).astype(np.int8)
    df['t_sin']   = np.sin(2 * np.pi * ts / 100).astype(np.float32)
    df['t_cos']   = np.cos(2 * np.pi * ts / 100).astype(np.float32)
    df['t_sin2']  = np.sin(2 * np.pi * ts / 52).astype(np.float32)
    df['t_cos2']  = np.cos(2 * np.pi * ts / 52).astype(np.float32)
    df['ts_norm'] = (ts / 4000.0).astype(np.float32)
    if 'horizon' in df.columns:
        df['horizon_log'] = np.log1p(df['horizon']).astype(np.float32)

    # 7. Lifecycle
    df = df.sort_values(GROUP_COLS + ['ts_index'])
    df['obs_idx']          = df.groupby(GROUP_COLS).cumcount().astype(np.int32)
    first_t                = df.groupby(GROUP_COLS)['ts_index'].transform('min')
    df['time_since_start'] = (df['ts_index'] - first_t).astype(np.int32)
    del first_t

    # 8. Lags, Rolling mean/std, EWM, Diff (NO median/max/min)
    key_present   = [f for f in KEY_FEATURES if E(f)]
    extra_present = [f for f in EXTRA_FEATURES if E(f)]
    inter_feats   = [f for f in ['d_al_am', 'd_cg_by', 'd_s_t',
                                  'mean_cg_by', 'd_al_bp', 'wap'] if E(f)]

    grouped = df.groupby(GROUP_COLS, sort=False)
    new = {}

    for col in key_present:
        for lag in LAG_STEPS:
            new[f'{col}_lag{lag}'] = grouped[col].shift(lag).astype(np.float32)
        for w in ROLLING_WINDOWS:
            shifted = grouped[col].shift(1)
            new[f'{col}_rmean_{w}'] = shifted.rolling(w, min_periods=1).mean().values.astype(np.float32)
            new[f'{col}_rstd_{w}']  = shifted.rolling(w, min_periods=1).std().values.astype(np.float32)
        new[f'{col}_ewm10'] = grouped[col].shift(1).ewm(span=10, min_periods=1).mean().values.astype(np.float32)
        new[f'{col}_diff1'] = grouped[col].diff(1).astype(np.float32)

    for col in extra_present:
        for lag in [1, 3, 5]:
            new[f'{col}_lag{lag}'] = grouped[col].shift(lag).astype(np.float32)
        new[f'{col}_diff1']   = grouped[col].diff(1).astype(np.float32)
        shifted = grouped[col].shift(1)
        new[f'{col}_rmean_5'] = shifted.rolling(5, min_periods=1).mean().values.astype(np.float32)
        new[f'{col}_ewm10']   = grouped[col].shift(1).ewm(span=10, min_periods=1).mean().values.astype(np.float32)

    for col in inter_feats:
        for lag in [1, 3]:
            new[f'{col}_lag{lag}'] = grouped[col].shift(lag).astype(np.float32)
        new[f'{col}_diff1'] = grouped[col].diff(1).astype(np.float32)

    if new:
        df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # 9. Momentum
    for col in key_present:
        l1, l5, rm5 = f'{col}_lag1', f'{col}_lag5', f'{col}_rmean_5'
        if E(l1) and E(l5):
            df[f'{col}_mom15'] = (df[l1] - df[l5]).astype(np.float32)
        if E(l1) and E(rm5):
            df[f'{col}_dev5']  = (df[l1] - df[rm5]).astype(np.float32)

    # 11. Pseudo-target features (from notebooks: feature_al is #1 strongest signal)
    # feature_al shifted by -h = value from h steps ago = causal "pseudo-target"
    if E('feature_al'):
        df['pseudo_al'] = grouped['feature_al'].shift(OPTIMAL_SHIFTS[horizon]).astype(np.float32)
    if E('feature_am'):
        df['pseudo_am'] = grouped['feature_am'].shift(OPTIMAL_SHIFTS[horizon]).astype(np.float32)
    if E('feature_cg'):
        df['pseudo_cg'] = grouped['feature_cg'].shift(OPTIMAL_SHIFTS[horizon]).astype(np.float32)
    if E('feature_s'):
        df['pseudo_s']  = grouped['feature_s'].shift(OPTIMAL_SHIFTS[horizon]).astype(np.float32)
    # Momentum of pseudo-targets
    if E('pseudo_al'):
        df['pseudo_al_diff'] = grouped['pseudo_al'].diff(1).astype(np.float32)

    # 12. Target lags (autocorrelation signal — from quantitive notebook)
    # CAUTION: past target is available in train rows, NaN in test rows.
    # For test, these will be 0.0 after fillna(0) — LGB can learn to ignore them.
    if TARGET in df.columns:
        df['y_lag1'] = grouped[TARGET].shift(1).astype(np.float32)
        df['y_lag3'] = grouped[TARGET].shift(3).astype(np.float32)
        df['y_diff1'] = grouped[TARGET].diff(1).astype(np.float32)
    # Expanding mean of target up to (but not including) current row
    if TARGET in df.columns:
        df['_cumsum'] = grouped[TARGET].cumsum().shift(1)
        df['_cumcnt'] = grouped[TARGET].cumcount()
        df['y_expand_mean'] = (df['_cumsum'] / (df['_cumcnt'] + 1e-9)).astype(np.float32)
        df.drop(columns=['_cumsum', '_cumcnt'], inplace=True)

    # 10. Fill NaN/Inf
    preserved_y = df[TARGET].copy() if TARGET in df.columns else None
    preserved_w = df[WEIGHT].copy() if WEIGHT in df.columns else None
    df = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    if preserved_y is not None: df[TARGET] = preserved_y
    if preserved_w is not None: df[WEIGHT] = preserved_w

    gc.collect()
    return df


def get_feature_columns(df):
    exclude = {'id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', WEIGHT, TARGET}
    return [c for c in df.columns if c not in exclude]


# ══════════════════════════════════════════════════════════════════
# SOLVER
# ══════════════════════════════════════════════════════════════════
def find_best_scale(y_va, val_pred, w_va, grid):
    best_s, best_score = 1.0, -1
    for s in grid:
        scaled = val_pred * s
        sc = weighted_rmse_score(y_va, scaled, w_va)
        if sc > best_score:
            best_score = sc
            best_s = s
    return best_s, best_score


def solve_horizon(horizon):
    t0 = time.time()
    print(f'\n{"="*60}')
    print(f'HORIZON {horizon}')
    print(f'{"="*60}')

    tr = pd.read_parquet(TRAIN_PATH).query('horizon == @horizon').reset_index(drop=True)
    te = pd.read_parquet(TEST_PATH).query('horizon == @horizon').reset_index(drop=True)
    print(f'Data: train={len(tr):,}, test={len(te):,}')

    target_stats = compute_target_stats(tr)

    te_tmp = te.copy()
    te_tmp[TARGET] = np.nan
    combined = pd.concat([tr, te_tmp], ignore_index=True)
    combined = combined.sort_values(GROUP_COLS + ['ts_index']).reset_index(drop=True)

    freq_stats = compute_freq_encoding(combined)

    print('Building features...')
    combined_feat = build_features(combined, target_stats, freq_stats)

    is_train = combined_feat[TARGET].notna()
    all_feat = combined_feat[is_train].reset_index(drop=True)
    te_feat  = combined_feat[~is_train].reset_index(drop=True)
    print(f'Split: train={len(all_feat):,}, test={len(te_feat):,}')
    del combined, combined_feat, te_tmp
    gc.collect()

    feats = get_feature_columns(all_feat)
    for c in feats:
        if c not in te_feat.columns:
            te_feat[c] = 0.0
    print(f'Features: {len(feats)}')

    # Train / Val split
    tr_mask = all_feat['ts_index'] <= VAL_THRESHOLD
    va_mask = all_feat['ts_index'] >  VAL_THRESHOLD
    X_tr = all_feat.loc[tr_mask, feats]
    y_tr = all_feat.loc[tr_mask, TARGET]
    w_tr = all_feat.loc[tr_mask, WEIGHT]
    X_va = all_feat.loc[va_mask, feats]
    y_va = all_feat.loc[va_mask, TARGET]
    w_va = all_feat.loc[va_mask, WEIGHT]
    print(f'Train: {len(X_tr):,}, Val: {len(X_va):,}')

    # Probe — find best_iter + feature importance
    probe = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=5000, random_state=42)
    probe.fit(
        X_tr, y_tr, sample_weight=w_tr,
        eval_set=[(X_va, y_va)], eval_sample_weight=[w_va],
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )
    best_iter = probe.best_iteration_ if probe.best_iteration_ else 3000
    best_iter = max(best_iter, 50)
    val_pred = probe.predict(X_va)

    # Per-horizon scale grid search
    best_scale, val_score_scaled = find_best_scale(y_va.values, val_pred, w_va.values, SCALE_GRID)
    val_score_raw = weighted_rmse_score(y_va.values, val_pred, w_va.values)
    print(f'Val WRMSE: {val_score_raw:.6f} (raw), {val_score_scaled:.6f} (scale={best_scale})')
    print(f'best_iter={best_iter}')

    # Feature pruning — keep top N by importance
    if TOP_FEATURES and TOP_FEATURES < len(feats):
        imp = pd.Series(probe.feature_importances_, index=feats).sort_values(ascending=False)
        pruned_feats = imp.head(TOP_FEATURES).index.tolist()
        print(f'Pruning: {len(feats)} → {len(pruned_feats)} features')
        feats = pruned_feats
    del probe
    gc.collect()

    # Retrain on ALL data with pruned features
    X_all = all_feat[feats]
    y_all = all_feat[TARGET]
    w_all = all_feat[WEIGHT]

    test_pred = np.zeros(len(te_feat), dtype=np.float64)
    for i, seed in enumerate(SEEDS):
        if i == 0 or (i + 1) % 5 == 0 or (i + 1) == N_SEEDS:
            print(f'  Seed {i+1}/{N_SEEDS}...')
        mdl = lgb.LGBMRegressor(**{**LGB_PARAMS, 'n_estimators': best_iter}, random_state=seed)
        mdl.fit(X_all, y_all, sample_weight=w_all)
        test_pred += mdl.predict(te_feat[feats]) / N_SEEDS
        del mdl
        gc.collect()

    # Post-processing: per-horizon scale + clip
    test_pred_scaled = test_pred * best_scale
    clip_lo = np.percentile(y_all.values, 0.5)
    clip_hi = np.percentile(y_all.values, 99.5)
    test_pred_final = np.clip(test_pred_scaled, clip_lo, clip_hi)
    print(f'Scale: ×{best_scale}, Clip: [{clip_lo:.2f}, {clip_hi:.2f}]')

    elapsed = (time.time() - t0) / 60
    print(f'H{horizon} done in {elapsed:.1f} min')

    test_ids = te_feat['id'].values
    del all_feat, te_feat, X_tr, X_va, X_all
    gc.collect()

    return {
        'horizon':    horizon,
        'ids':        test_ids,
        'pred':       test_pred_final,
        'pred_raw':   test_pred,
        'val_score':  val_score_scaled,
        'val_score_raw': val_score_raw,
        'best_scale': best_scale,
        'val_y':      y_va.values,
        'val_w':      w_va.values,
        'val_pred':   val_pred,
        'best_iter':  best_iter,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print('=' * 60)
    print(f'V4 — ENHANCED PIPELINE')
    print(f'Config: {N_SEEDS} seeds, top {TOP_FEATURES} feats, per-horizon scale')
    print(f'       + pseudo-target features + target lag features')
    print('=' * 60)

    results = []
    for h in HORIZONS:
        results.append(solve_horizon(h))

    # Build submission
    sub_parts = []
    for r in results:
        sub_parts.append(pd.DataFrame({'id': r['ids'], 'prediction': r['pred']}))
    sub = pd.concat(sub_parts, ignore_index=True)

    # Aggregate score (each horizon's val_pred is raw; aggregate raw → comparable across runs)
    all_y = np.concatenate([r['val_y'] for r in results])
    all_w = np.concatenate([r['val_w'] for r in results])
    all_p_raw = np.concatenate([r['val_pred'] for r in results])
    agg_score_raw = weighted_rmse_score(all_y, all_p_raw, all_w)
    # Also compute with per-horizon scales applied for reference
    all_p_scaled = np.concatenate([r['val_pred'] * r['best_scale'] for r in results])
    agg_score_scaled = weighted_rmse_score(all_y, all_p_scaled, all_w)

    print('\n' + '=' * 60)
    print('RESULTS')
    for r in sorted(results, key=lambda x: x['horizon']):
        print(f"  H{r['horizon']:>2}: val={r['val_score']:.6f} (scale={r['best_scale']}), iter={r['best_iter']}")
    print(f'  Aggregate raw:    {agg_score_raw:.6f}')
    print(f'  Aggregate scaled: {agg_score_scaled:.6f}')
    print('=' * 60)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub[["id", "prediction"]].to_csv(path, index=False)
    print(f'\nSaved: {path} ({len(sub):,} rows)')

    # Save raw predictions for postprocess.py
    raw_parts = []
    for r in results:
        raw_parts.append(pd.DataFrame({
            'id': r['ids'],
            'pred_raw': r['pred_raw'],
            'horizon': r['horizon'],
        }))
    raw_df = pd.concat(raw_parts, ignore_index=True)
    raw_path = os.path.join(OUTPUT_DIR, "raw_predictions.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f'Raw predictions: {raw_path}')
    print('\n🚀 DONE — V4 Enhanced Pipeline')
