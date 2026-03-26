import lightgbm as lgb
import numpy as np
import pandas as pd
import gc
import time
from sklearn.linear_model import LinearRegression
from config import (
    LGB_PARAMS, TARGET, WEIGHT, HORIZONS,
    N_SEEDS, SEEDS, VAL_THRESHOLD,
    TRAIN_PATH, TEST_PATH, GROUP_COLS
)
from evaluation import weighted_rmse_score
from features import compute_target_stats, compute_freq_encoding, build_features, get_feature_columns


def solve_horizon(horizon):
    t0 = time.time()
    print(f'\n{"="*60}')
    print(f'HORIZON {horizon}')
    print(f'{"="*60}')

    # ── Load data ──
    tr = pd.read_parquet(TRAIN_PATH).query('horizon == @horizon').reset_index(drop=True)
    te = pd.read_parquet(TEST_PATH).query('horizon == @horizon').reset_index(drop=True)
    print(f'Data: train={len(tr):,}, test={len(te):,}')

    # ── Stats from train only ──
    target_stats = compute_target_stats(tr)

    # ── Concat train+test for continuous lags ──
    te_tmp = te.copy()
    te_tmp[TARGET] = np.nan
    combined = pd.concat([tr, te_tmp], ignore_index=True)
    combined = combined.sort_values(GROUP_COLS + ['ts_index']).reset_index(drop=True)

    # Freq encoding on combined data (no target used → rule-safe)
    freq_stats = compute_freq_encoding(combined)

    print('Building features...')
    combined_feat = build_features(combined, target_stats, freq_stats)

    is_train = combined_feat[TARGET].notna()
    all_feat = combined_feat[is_train].reset_index(drop=True)
    te_feat = combined_feat[~is_train].reset_index(drop=True)
    print(f'Split: train_feat={len(all_feat):,}, test_feat={len(te_feat):,}')
    del combined, combined_feat, te_tmp
    gc.collect()

    feats = get_feature_columns(all_feat)
    for c in feats:
        if c not in te_feat.columns:
            te_feat[c] = 0.0
    print(f'Features: {len(feats)}')

    # ── Train/Val split ──
    tr_mask = all_feat['ts_index'] <= VAL_THRESHOLD
    va_mask = all_feat['ts_index'] > VAL_THRESHOLD

    X_tr = all_feat.loc[tr_mask, feats]
    y_tr = all_feat.loc[tr_mask, TARGET]
    w_tr = all_feat.loc[tr_mask, WEIGHT]
    X_va = all_feat.loc[va_mask, feats]
    y_va = all_feat.loc[va_mask, TARGET]
    w_va = all_feat.loc[va_mask, WEIGHT]
    print(f'Train: {len(X_tr):,}, Val: {len(X_va):,}')

    # ── Find best iteration ──
    probe = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=5000, random_state=42)
    probe.fit(
        X_tr, y_tr, sample_weight=w_tr,
        eval_set=[(X_va, y_va)], eval_sample_weight=[w_va],
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )
    best_iter = probe.best_iteration_ if probe.best_iteration_ else 3000
    best_iter = max(best_iter, 50)

    # ── Validation score ──
    val_pred = probe.predict(X_va)
    val_score_raw = weighted_rmse_score(y_va.values, val_pred, w_va.values)
    print(f'Val score (probe): {val_score_raw:.6f}, best_iter={best_iter}')
    del probe
    gc.collect()

    # ── Calibration on validation ──
    calibrator = LinearRegression()
    calibrator.fit(val_pred.reshape(-1, 1), y_va.values, sample_weight=w_va.values)
    cal_val = calibrator.predict(val_pred.reshape(-1, 1)).ravel()
    val_score_cal = weighted_rmse_score(y_va.values, cal_val, w_va.values)
    print(f'Calibration: a={calibrator.coef_[0]:.4f}, b={calibrator.intercept_:.6f}')
    print(f'Val score (calibrated): {val_score_cal:.6f}')

    # ── Retrain on ALL data with N seeds ──
    X_all = all_feat[feats]
    y_all = all_feat[TARGET]
    w_all = all_feat[WEIGHT]

    test_pred = np.zeros(len(te_feat), dtype=np.float64)
    for i, seed in enumerate(SEEDS):
        if i == 0 or (i + 1) % 5 == 0:
            print(f'  Seed {i+1}/{N_SEEDS}...')
        mdl = lgb.LGBMRegressor(**{**LGB_PARAMS, 'n_estimators': best_iter}, random_state=seed)
        mdl.fit(X_all, y_all, sample_weight=w_all)
        test_pred += mdl.predict(te_feat[feats]) / N_SEEDS
        del mdl
        gc.collect()

    # ── Apply calibration to test predictions ──
    test_pred_cal = calibrator.predict(test_pred.reshape(-1, 1)).ravel()

    # ── Post-processing: percentile clip ──
    clip_lo = np.percentile(y_all.values, 0.5)
    clip_hi = np.percentile(y_all.values, 99.5)
    test_pred_clip = np.clip(test_pred_cal, clip_lo, clip_hi)
    print(f'Clip range: [{clip_lo:.2f}, {clip_hi:.2f}]')

    elapsed = (time.time() - t0) / 60
    print(f'Horizon {horizon} done in {elapsed:.1f} min')

    test_ids = te_feat['id'].values

    del all_feat, te_feat, X_tr, X_va, X_all
    gc.collect()

    return {
        'horizon': horizon,
        'ids': test_ids,
        'pred_raw': test_pred,
        'pred_clip': test_pred_clip,
        'val_score_raw': val_score_raw,
        'val_score_cal': val_score_cal,
        'val_y': y_va.values,
        'val_w': w_va.values,
        'val_pred': cal_val,
    }


def train_and_predict_all_horizons():
    print('=' * 60)
    print('V6 HYBRID — LGB + Calibration + Concat + FreqEnc')
    print('=' * 60)
    print(f'Config: {N_SEEDS} seeds, val_split={VAL_THRESHOLD}')
    print(f'Horizons: {HORIZONS}')

    results = []
    for h in HORIZONS:
        results.append(solve_horizon(h))

    # ── Build submissions ──
    sub_parts = []
    sub_raw_parts = []
    for r in results:
        sub_parts.append(pd.DataFrame({'id': r['ids'], 'prediction': r['pred_clip']}))
        sub_raw_parts.append(pd.DataFrame({'id': r['ids'], 'prediction': r['pred_raw']}))

    sub = pd.concat(sub_parts, ignore_index=True)
    sub_raw = pd.concat(sub_raw_parts, ignore_index=True)

    # ── Aggregate score ──
    all_y = np.concatenate([r['val_y'] for r in results])
    all_w = np.concatenate([r['val_w'] for r in results])
    all_p = np.concatenate([r['val_pred'] for r in results])
    agg_score = weighted_rmse_score(all_y, all_p, all_w)

    print('\n' + '=' * 60)
    print('RESULTS')
    print(f'{"H":>4} | {"Raw":>10} | {"Calibrated":>10}')
    print('-' * 30)
    for r in sorted(results, key=lambda x: x['horizon']):
        print(f"  {r['horizon']:>2} | {r['val_score_raw']:.6f} | {r['val_score_cal']:.6f}")
    print('-' * 30)
    print(f'  Aggregate (calibrated): {agg_score:.6f}')
    print('=' * 60)

    return sub, sub_raw, {r['horizon']: r['val_score_cal'] for r in results}


def create_submission(submission_df, filename="submission.csv"):
    from config import OUTPUT_DIR
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    submission_df[["id", "prediction"]].to_csv(path, index=False)
    print(f"Saved: {path} ({submission_df.shape[0]:,} rows)")
    print(submission_df.head())
    return path
