import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import gc
import time
from config import (
    LGBM_BASE_PARAMS, XGB_BASE_PARAMS, TARGET, WEIGHT, HORIZONS,
    SEEDS, XGB_SEEDS, TRAIN_PATH, VAL_THRESHOLD,
    CLIP_Q_LOW, CLIP_Q_HIGH, BLEND_W_LGB, BLEND_W_XGB,
)
from evaluation import weighted_rmse_score
from features import compute_target_encoding_stats, build_features, get_feature_columns


def _train_lgb_ensemble(X_fit, y_fit, w_fit, X_val, y_val, w_val,
                        X_all, y_all, w_all, X_test, feats):
    """Train LightGBM multi-seed ensemble with retrain-on-all."""
    val_pred = np.zeros(len(y_val), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    best_iters = []

    n_seeds = len(SEEDS)
    for i, seed in enumerate(SEEDS, 1):
        if i == 1 or i % 5 == 0:
            print(f'    [LGB] Seed {i}/{n_seeds} ...')

        mdl = lgb.LGBMRegressor(
            **LGBM_BASE_PARAMS,
            n_estimators=5000,
            random_state=seed,
        )
        mdl.fit(
            X_fit, y_fit,
            sample_weight=w_fit,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )

        bi = mdl.best_iteration_ if mdl.best_iteration_ is not None else 5000
        bi = max(int(bi), 20)
        best_iters.append(bi)

        val_pred += mdl.predict(X_val) / n_seeds

        # Retrain on all data
        mdl_full = lgb.LGBMRegressor(
            **{**LGBM_BASE_PARAMS, 'n_estimators': bi},
            random_state=seed,
        )
        mdl_full.fit(X_all, y_all, sample_weight=w_all)
        test_pred += mdl_full.predict(X_test[feats]) / n_seeds

        del mdl, mdl_full
        gc.collect()

    print(f'    [LGB] avg_iter={np.mean(best_iters):.0f}')
    return val_pred, test_pred


def _train_xgb_ensemble(X_fit, y_fit, w_fit, X_val, y_val, w_val,
                         X_all, y_all, w_all, X_test, feats):
    """Train XGBoost multi-seed ensemble with retrain-on-all."""
    val_pred = np.zeros(len(y_val), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    best_iters = []

    n_seeds = len(XGB_SEEDS)
    for i, seed in enumerate(XGB_SEEDS, 1):
        if i == 1 or i % 5 == 0:
            print(f'    [XGB] Seed {i}/{n_seeds} ...')

        mdl = xgb.XGBRegressor(
            **XGB_BASE_PARAMS,
            n_estimators=5000,
            random_state=seed,
        )
        mdl.fit(
            X_fit, y_fit,
            sample_weight=w_fit,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False,
        )

        bi = mdl.best_iteration
        if bi is None or bi < 20:
            bi = 5000
        best_iters.append(bi)

        val_pred += mdl.predict(X_val) / n_seeds

        # Retrain on all data
        mdl_full = xgb.XGBRegressor(
            **{**XGB_BASE_PARAMS, 'n_estimators': bi},
            random_state=seed,
        )
        mdl_full.fit(X_all, y_all, sample_weight=w_all, verbose=False)
        test_pred += mdl_full.predict(X_test[feats]) / n_seeds

        del mdl, mdl_full
        gc.collect()

    print(f'    [XGB] avg_iter={np.mean(best_iters):.0f}')
    return val_pred, test_pred


def solve_horizon(horizon, target_stats):
    t_h = time.time()
    print()
    print('=' * 70)
    print(f'HORIZON {horizon}')
    print('=' * 70)

    # ── 1. Load & build features ──
    tr = pd.read_parquet(TRAIN_PATH).query('horizon == @horizon')

    from config import TEST_PATH
    te = pd.read_parquet(TEST_PATH).query('horizon == @horizon')

    print(f'Data: train={len(tr):,}, test={len(te):,}')

    print('  Building features...')
    tr = build_features(tr, target_stats)
    te = build_features(te, target_stats)

    feats = get_feature_columns(tr)
    for c in feats:
        if c not in te.columns:
            te[c] = 0.0

    print(f'  Features: {len(feats)}')

    # ── 2. Split ──
    fit_m = tr.ts_index <= VAL_THRESHOLD
    val_m = tr.ts_index > VAL_THRESHOLD

    X_fit = tr.loc[fit_m, feats]
    y_fit = tr.loc[fit_m, TARGET]
    w_fit = tr.loc[fit_m, WEIGHT]

    X_val = tr.loc[val_m, feats]
    y_val = tr.loc[val_m, TARGET]
    w_val = tr.loc[val_m, WEIGHT]

    X_all = tr[feats]
    y_all = tr[TARGET]
    w_all = tr[WEIGHT]

    print(f'  Train: {X_fit.shape[0]:,}, Val: {X_val.shape[0]:,}, All: {X_all.shape[0]:,}')

    # ── 3. Train LightGBM ensemble ──
    print('  Training LightGBM...')
    lgb_val, lgb_test = _train_lgb_ensemble(
        X_fit, y_fit, w_fit, X_val, y_val, w_val,
        X_all, y_all, w_all, te, feats,
    )

    # ── 4. Train XGBoost ensemble ──
    print('  Training XGBoost...')
    xgb_val, xgb_test = _train_xgb_ensemble(
        X_fit, y_fit, w_fit, X_val, y_val, w_val,
        X_all, y_all, w_all, te, feats,
    )

    # ── 5. Blend ──
    val_pred = BLEND_W_LGB * lgb_val + BLEND_W_XGB * xgb_val
    test_pred = BLEND_W_LGB * lgb_test + BLEND_W_XGB * xgb_test

    # Individual scores
    score_lgb = weighted_rmse_score(y_val.values, lgb_val, w_val.values)
    score_xgb = weighted_rmse_score(y_val.values, xgb_val, w_val.values)
    score_blend = weighted_rmse_score(y_val.values, val_pred, w_val.values)

    print(f'  LGB={score_lgb:.6f} | XGB={score_xgb:.6f} | Blend={score_blend:.6f}')

    # ── 6. Clip predictions ──
    q_low, q_high = np.quantile(y_fit.values, [CLIP_Q_LOW, CLIP_Q_HIGH])
    test_pred_clip = np.clip(test_pred, q_low, q_high)

    elapsed = (time.time() - t_h) / 60
    print(f'  Total: {elapsed:.1f} min')

    out = {
        'horizon': horizon,
        'id_test': te['id'].values,
        'test_pred_raw': test_pred,
        'test_pred_clip': test_pred_clip,
        'y_val': y_val.values,
        'w_val': w_val.values,
        'val_pred': val_pred,
        'score_local': score_blend,
        'score_lgb': score_lgb,
        'score_xgb': score_xgb,
    }

    del tr, te, X_fit, y_fit, w_fit, X_val, y_val, w_val, X_all, y_all, w_all
    gc.collect()

    return out


def train_and_predict_all_horizons():
    print("=" * 70)
    print("Computing target encoding stats (fit split only)...")
    print("=" * 70)
    target_stats = compute_target_encoding_stats(TRAIN_PATH)
    print("Stats ready.")
    print(f"Blend weights: LGB={BLEND_W_LGB}, XGB={BLEND_W_XGB}")

    all_outputs = []
    for h in HORIZONS:
        all_outputs.append(solve_horizon(h, target_stats))

    # ── Build submissions ──
    sub_clip_parts = []
    sub_raw_parts = []
    oof_parts = []

    for out in all_outputs:
        sub_clip_parts.append(pd.DataFrame({
            'id': out['id_test'],
            'prediction': out['test_pred_clip'],
        }))
        sub_raw_parts.append(pd.DataFrame({
            'id': out['id_test'],
            'prediction': out['test_pred_raw'],
        }))
        oof_parts.append(pd.DataFrame({
            'y_true': out['y_val'],
            'y_pred': out['val_pred'],
            'w': out['w_val'],
            'horizon': out['horizon'],
        }))

    sub_clip = pd.concat(sub_clip_parts, ignore_index=True)
    sub_raw = pd.concat(sub_raw_parts, ignore_index=True)
    oof = pd.concat(oof_parts, ignore_index=True)

    # ── Aggregate score ──
    agg_score = weighted_rmse_score(oof['y_true'].values, oof['y_pred'].values, oof['w'].values)

    print()
    print('=' * 70)
    print('PER-HORIZON LOCAL SCORES')
    print(f'{"H":>4} | {"LGB":>10} | {"XGB":>10} | {"Blend":>10}')
    print('-' * 42)
    for out in sorted(all_outputs, key=lambda d: d['horizon']):
        print(f"  {out['horizon']:>2} | {out['score_lgb']:.6f} | {out['score_xgb']:.6f} | {out['score_local']:.6f}")
    print('-' * 42)
    print(f'  Aggregate (blend) local score: {agg_score:.6f}')
    print('=' * 70)

    return sub_clip, sub_raw, {out['horizon']: out['score_local'] for out in all_outputs}


def create_submission(submission_df, filename="submission.csv"):
    from config import OUTPUT_DIR
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    submission_df[["id", "prediction"]].to_csv(path, index=False)
    print(f"Saved: {path} ({submission_df.shape[0]} rows)")
    print(submission_df.head())
    return path
