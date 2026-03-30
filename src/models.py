import lightgbm as lgb
import numpy as np
import pandas as pd
import gc
import time
from config import (
    LGB_PARAMS, TARGET, WEIGHT, HORIZONS,
    N_SEEDS, SEEDS, VAL_THRESHOLD,
    TRAIN_PATH, TEST_PATH, GROUP_COLS,
    OPTIMAL_SHIFTS, USE_RECENCY_WEIGHTING, RECENCY_FACTOR,
    SCALE_GRID, TOP_FEATURES,
)
from evaluation import weighted_rmse_score
from features import compute_target_stats, compute_freq_encoding, build_features, get_feature_columns


# ══════════════════════════════════════════════════════════════════
# CUSTOM SKILL METRIC
# ══════════════════════════════════════════════════════════════════
def lgb_skill_metric(y_pred, dataset):
    y_true = dataset.get_label()
    w = dataset.get_weight()
    if w is None:
        w = np.ones_like(y_true, dtype=np.float64)
    return "skill_score", weighted_rmse_score(y_true, y_pred, w), True


# ══════════════════════════════════════════════════════════════════
# SCALE OPTIMIZATION
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


# ══════════════════════════════════════════════════════════════════
# SOLVER
# ══════════════════════════════════════════════════════════════════
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
    # build_features now accepts `horizon` for pseudo-target shifting
    combined_feat = build_features(combined, target_stats, freq_stats, horizon)

    is_train = combined_feat[TARGET].notna()
    all_feat = combined_feat[is_train].reset_index(drop=True)
    te_feat  = combined_feat[~is_train].reset_index(drop=True)
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
    va_mask = all_feat['ts_index'] >  VAL_THRESHOLD
    X_tr = all_feat.loc[tr_mask, feats]
    y_tr = all_feat.loc[tr_mask, TARGET]
    w_tr = all_feat.loc[tr_mask, WEIGHT]
    X_va = all_feat.loc[va_mask, feats]
    y_va = all_feat.loc[va_mask, TARGET]
    w_va = all_feat.loc[va_mask, WEIGHT]
    print(f'Train: {len(X_tr):,}, Val: {len(X_va):,}')

    # ── Native categorical columns ──
    cat_cols = [c for c in ['code', 'sub_code', 'sub_category'] if c in feats]

    # ── Probe: find best_iter via LGB native API + custom feval ──
    print('Probe training with custom skill metric...')
    d_tr = lgb.Dataset(
        X_tr, label=y_tr, weight=w_tr,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    d_va = lgb.Dataset(
        X_va, label=y_va, weight=w_va,
        categorical_feature=cat_cols,
        reference=d_tr,
        free_raw_data=False,
    )
    probe_model = lgb.train(
        {**LGB_PARAMS, 'n_estimators': 5000, 'seed': 42},
        d_tr,
        valid_sets=[d_va],
        valid_names=['valid'],
        feval=lgb_skill_metric,
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    best_iter = probe_model.best_iteration if probe_model.best_iteration else 3000
    best_iter = max(best_iter, 50)
    val_pred = probe_model.predict(X_va)
    del probe_model, d_tr, d_va
    gc.collect()

    # ── Per-horizon scale grid search ──
    best_scale, val_score_scaled = find_best_scale(y_va.values, val_pred, w_va.values, SCALE_GRID)
    val_score_raw = weighted_rmse_score(y_va.values, val_pred, w_va.values)
    print(f'Val WRMSE: {val_score_raw:.6f} (raw), {val_score_scaled:.6f} (scale={best_scale})')
    print(f'best_iter={best_iter}')

    # ── Feature pruning — keep top N by importance ──
    print('Feature pruning...')
    d_tr_imp = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                           categorical_feature=cat_cols, free_raw_data=False)
    d_va_imp = lgb.Dataset(X_va, label=y_va, weight=w_va,
                           categorical_feature=cat_cols, reference=d_tr_imp, free_raw_data=False)
    imp_model = lgb.train(
        {**LGB_PARAMS, 'n_estimators': best_iter, 'seed': 42},
        d_tr_imp, valid_sets=[d_va_imp], valid_names=['valid'],
        callbacks=[lgb.log_evaluation(0)],
    )
    imp = pd.Series(imp_model.feature_importance(importance_type='gain'), index=feats)
    pruned_feats = imp.sort_values(ascending=False).head(TOP_FEATURES).index.tolist()
    print(f'Pruning: {len(feats)} → {len(pruned_feats)} features')
    feats = pruned_feats
    pruned_cat_cols = [c for c in cat_cols if c in feats]
    del imp_model, d_tr_imp, d_va_imp
    gc.collect()

    # ── Retrain on ALL data with pruned features ──
    X_all = all_feat[feats].copy()
    y_all = all_feat[TARGET].values
    w_all_orig = all_feat[WEIGHT].values.copy()

    # ── Linear recency weighting ──
    if USE_RECENCY_WEIGHTING:
        ts_min = float(all_feat['ts_index'].min())
        ts_max = float(all_feat['ts_index'].max())
        recency = (all_feat['ts_index'].values - ts_min) / (ts_max - ts_min + 1e-9)
        w_all = w_all_orig * (0.5 + RECENCY_FACTOR * recency.astype(np.float64))
        print(f'Recency weighting: factor range {w_all.min()/w_all_orig.min():.1f}–{w_all.max()/w_all_orig.max():.1f}')
    else:
        w_all = w_all_orig

    # ── Retrain with multiple seeds ──
    test_pred = np.zeros(len(te_feat), dtype=np.float64)
    for i, seed in enumerate(SEEDS):
        if i == 0 or (i + 1) % 5 == 0 or (i + 1) == N_SEEDS:
            print(f'  Seed {i+1}/{N_SEEDS}...')
        d_full = lgb.Dataset(X_all, label=y_all, weight=w_all,
                             categorical_feature=pruned_cat_cols, free_raw_data=False)
        mdl = lgb.train(
            {**LGB_PARAMS, 'n_estimators': best_iter, 'seed': seed},
            d_full,
            callbacks=[lgb.log_evaluation(0)],
        )
        X_te = te_feat[feats].copy()
        for c in pruned_cat_cols:
            X_te[c] = X_te[c].astype('category')
            X_te[c] = X_te[c].cat.set_categories(X_all[c].cat.categories)
        test_pred += mdl.predict(X_te) / N_SEEDS
        del mdl, d_full, X_te
        gc.collect()

    # ── Post-processing: per-horizon scale + clip ──
    test_pred_scaled = test_pred * best_scale
    clip_lo = np.percentile(y_all, 0.5)
    clip_hi = np.percentile(y_all, 99.5)
    test_pred_final = np.clip(test_pred_scaled, clip_lo, clip_hi)
    print(f'Scale: x{best_scale}, Clip: [{clip_lo:.2f}, {clip_hi:.2f}]')

    elapsed = (time.time() - t0) / 60
    print(f'H{horizon} done in {elapsed:.1f} min')

    test_ids = te_feat['id'].values
    del all_feat, te_feat, X_tr, X_va, X_all
    gc.collect()

    return {
        'horizon':       horizon,
        'ids':           test_ids,
        'pred':          test_pred_final,
        'pred_raw':      test_pred,
        'val_score':     val_score_scaled,
        'val_score_raw': val_score_raw,
        'best_scale':    best_scale,
        'val_y':         y_va.values,
        'val_w':         w_va.values,
        'val_pred':      val_pred,
        'best_iter':     best_iter,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def train_and_predict_all_horizons():
    print('=' * 60)
    print(f'V4 ENHANCED — LGB Native API + Skill Metric + Scale Opt')
    print(f'Config: {N_SEEDS} seeds, top {TOP_FEATURES} feats, per-horizon scale')
    print(f'       + pseudo-target + target lags + recency weighting + native cats')
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

    return sub, {r['horizon']: r['val_score'] for r in results}


def create_submission(submission_df, filename="submission.csv"):
    from config import OUTPUT_DIR
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    submission_df[["id", "prediction"]].to_csv(path, index=False)
    print(f"Saved: {path} ({submission_df.shape[0]:,} rows)")
    return path
