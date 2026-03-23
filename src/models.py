import lightgbm as lgb
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import gc
import time
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from config import (
    LGBM_BASE_PARAMS, CATBOOST_PARAMS,
    TARGET, WEIGHT, HORIZONS,
    CV_LGB_SEEDS, CV_CAT_SEEDS,
    FINAL_LGB_SEEDS, FINAL_CAT_SEEDS,
    TRAIN_PATH, N_CV_SPLITS
)
from evaluation import weighted_rmse_score
from features import compute_target_encoding_stats, build_features, get_feature_columns

def _train_cat_fold(X_tr, y_tr, w_tr, X_va, y_va, w_va, seeds):
    preds = np.zeros(len(X_va), dtype=np.float64)
    for seed in seeds:
        mdl = CatBoostRegressor(
            **{**CATBOOST_PARAMS, 'random_seed': seed},
        )
        mdl.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va))
        preds += mdl.predict(X_va) / len(seeds)
        del mdl
        gc.collect()
    return preds

def _train_lgb_final(X_all, y_all, w_all, X_test, feats,
                     X_fit, y_fit, w_fit, X_val, y_val, w_val):
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    seeds = FINAL_LGB_SEEDS
    n = len(seeds)
    best_iters = []

    for i, seed in enumerate(seeds, 1):
        if i == 1 or i % 5 == 0:
            print(f'    [LGB] Seed {i}/{n} ...')

        # Find best iteration on last fold
        probe = lgb.LGBMRegressor(**LGBM_BASE_PARAMS, n_estimators=5000, random_state=seed)
        probe.fit(
            X_fit, y_fit, sample_weight=w_fit,
            eval_set=[(X_val, y_val)], eval_sample_weight=[w_val],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        bi = probe.best_iteration_ if probe.best_iteration_ is not None else 5000
        bi = max(int(bi), 20)
        best_iters.append(bi)

        # Retrain on ALL data
        mdl = lgb.LGBMRegressor(**{**LGBM_BASE_PARAMS, 'n_estimators': bi}, random_state=seed)
        mdl.fit(X_all, y_all, sample_weight=w_all)
        test_pred += mdl.predict(X_test[feats]) / n

        del probe, mdl
        gc.collect()

    print(f'    [LGB] avg_iter={np.mean(best_iters):.0f}')
    return test_pred


def _train_cat_final(X_all, y_all, w_all, X_test, feats,
                     X_fit, y_fit, w_fit, X_val, y_val):
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    seeds = FINAL_CAT_SEEDS
    n = len(seeds)
    best_iters = []

    for i, seed in enumerate(seeds, 1):
        if i == 1 or i % 5 == 0:
            print(f'    [CAT] Seed {i}/{n} ...')

        # Find best iteration
        probe = CatBoostRegressor(**{**CATBOOST_PARAMS, 'random_seed': seed})
        probe.fit(X_fit, y_fit, sample_weight=w_fit, eval_set=(X_val, y_val))
        bi = probe.get_best_iteration()
        if bi is None or bi < 20:
            bi = 3000
        best_iters.append(bi)

        # Retrain on ALL data
        cat_params = {k: v for k, v in CATBOOST_PARAMS.items() if k != 'early_stopping_rounds'}
        mdl = CatBoostRegressor(**{**cat_params, 'iterations': bi, 'random_seed': seed})
        mdl.fit(X_all, y_all, sample_weight=w_all)
        test_pred += mdl.predict(X_test[feats]) / n

        del probe, mdl
        gc.collect()

    print(f'    [CAT] avg_iter={np.mean(best_iters):.0f}')
    return test_pred

def solve_horizon(horizon):
    t_h = time.time()
    print()
    print('=' * 70)
    print(f'HORIZON {horizon}')
    print('=' * 70)

    # ── 1. Load data ──
    tr = pd.read_parquet(TRAIN_PATH).query('horizon == @horizon')
    from config import TEST_PATH
    te = pd.read_parquet(TEST_PATH).query('horizon == @horizon')
    print(f'Data: train={len(tr):,}, test={len(te):,}')

    # ── 2. TimeSeriesSplit on unique timestamps ──
    unique_times = np.sort(tr['ts_index'].unique())
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    oof_lgb = np.full(len(tr), np.nan, dtype=np.float64)
    oof_cat = np.full(len(tr), np.nan, dtype=np.float64)
    oof_y = np.full(len(tr), np.nan, dtype=np.float64)
    oof_w = np.full(len(tr), np.nan, dtype=np.float64)
    oof_mask = np.zeros(len(tr), dtype=bool)

    last_fold_data = {}  # save last fold for final retrain iteration estimation

    for fold, (train_time_idx, val_time_idx) in enumerate(tscv.split(unique_times)):
        train_times = set(unique_times[train_time_idx])
        val_times = set(unique_times[val_time_idx])

        tr_mask = tr['ts_index'].isin(train_times)
        va_mask = tr['ts_index'].isin(val_times)

        tr_fold = tr[tr_mask].copy()
        va_fold = tr[va_mask].copy()

        print(f'\n  Fold {fold+1}/{N_CV_SPLITS}: train={len(tr_fold):,}, val={len(va_fold):,} '
              f'(t_train=[{min(train_times)}-{max(train_times)}], t_val=[{min(val_times)}-{max(val_times)}])')

        # ── Per-fold target encoding ──
        fold_stats = compute_target_encoding_stats(tr_fold)

        # ── Build features ──
        tr_feat = build_features(tr_fold, fold_stats)
        va_feat = build_features(va_fold, fold_stats)

        feats = get_feature_columns(tr_feat)
        for c in feats:
            if c not in va_feat.columns:
                va_feat[c] = 0.0

        X_tr = tr_feat[feats]
        y_tr = tr_feat[TARGET]
        w_tr = tr_feat[WEIGHT]
        X_va = va_feat[feats]
        y_va = va_feat[TARGET]
        w_va = va_feat[WEIGHT]

        # ── Train LGB on this fold ──
        print(f'    Training LGB ({len(CV_LGB_SEEDS)} seeds)...')
        lgb_pred = _train_lgb_fold_simple(X_tr, y_tr, w_tr, X_va, y_va, w_va, CV_LGB_SEEDS)

        # ── Train CatBoost on this fold ──
        print(f'    Training CAT ({len(CV_CAT_SEEDS)} seeds)...')
        cat_pred = _train_cat_fold(X_tr, y_tr, w_tr, X_va, y_va, w_va, CV_CAT_SEEDS)

        # ── Store OOF predictions ──
        va_indices = tr.index[va_mask]
        oof_lgb[va_indices] = lgb_pred
        oof_cat[va_indices] = cat_pred
        oof_y[va_indices] = y_va.values
        oof_w[va_indices] = w_va.values
        oof_mask[va_indices] = True

        # Save last fold for final retrain iteration estimation
        last_fold_data = {
            'X_fit': X_tr, 'y_fit': y_tr, 'w_fit': w_tr,
            'X_val': X_va, 'y_val': y_va, 'w_val': w_va,
            'feats': feats,
        }

        fold_score_lgb = weighted_rmse_score(y_va.values, lgb_pred, w_va.values)
        fold_score_cat = weighted_rmse_score(y_va.values, cat_pred, w_va.values)
        print(f'    Fold {fold+1} LGB={fold_score_lgb:.6f} | CAT={fold_score_cat:.6f}')

        del tr_feat, va_feat, X_tr, y_tr, w_tr, X_va, y_va, w_va, tr_fold, va_fold
        gc.collect()

    # ── 3. Ridge Stacking on all OOF ──
    print('\n  Ridge stacking on OOF...')
    valid = oof_mask
    oof_stack = np.column_stack([oof_lgb[valid], oof_cat[valid]])
    ridge = Ridge(alpha=1.0)
    ridge.fit(oof_stack, oof_y[valid], sample_weight=oof_w[valid])
    print(f'    Ridge weights: LGB={ridge.coef_[0]:.4f}, CAT={ridge.coef_[1]:.4f}, intercept={ridge.intercept_:.6f}')

    raw_oof_pred = ridge.predict(oof_stack)
    
    # ── 3.1. Post-Processing OOF (Neutralization, Clipping, Shrinkage) ──
    oof_pred = raw_oof_pred - np.mean(raw_oof_pred)
    oof_pred = np.clip(oof_pred, -0.02, 0.02)
    oof_pred = oof_pred * 0.9

    score_lgb = weighted_rmse_score(oof_y[valid], oof_lgb[valid], oof_w[valid])
    score_cat = weighted_rmse_score(oof_y[valid], oof_cat[valid], oof_w[valid])
    score_ridge = weighted_rmse_score(oof_y[valid], oof_pred, oof_w[valid])
    print(f'  OOF: LGB={score_lgb:.6f} | CAT={score_cat:.6f} | Ridge={score_ridge:.6f}')

    # ── 4. Final retrain on ALL data → predict test ──
    print('\n  Final retrain on ALL data...')
    all_stats = compute_target_encoding_stats(tr)
    tr_all = build_features(tr, all_stats)
    te_feat = build_features(te, all_stats)

    feats = get_feature_columns(tr_all)
    for c in feats:
        if c not in te_feat.columns:
            te_feat[c] = 0.0

    X_all = tr_all[feats]
    y_all = tr_all[TARGET]
    w_all = tr_all[WEIGHT]

    # Use last fold split for iteration estimation
    lfd = last_fold_data

    print('  Final LGB retrain...')
    lgb_test = _train_lgb_final(
        X_all, y_all, w_all, te_feat, feats,
        lfd['X_fit'], lfd['y_fit'], lfd['w_fit'],
        lfd['X_val'], lfd['y_val'], lfd['w_val'],
    )

    print('  Final CAT retrain...')
    cat_test = _train_cat_final(
        X_all, y_all, w_all, te_feat, feats,
        lfd['X_fit'], lfd['y_fit'], lfd['w_fit'],
        lfd['X_val'], lfd['y_val'],
    )

    # Apply Ridge to test predictions
    test_stack = np.column_stack([lgb_test, cat_test])
    test_pred = ridge.predict(test_stack)

    # ── 5. Post-Processing & Neutralization ──
    # 1. Khử Bias
    test_pred_neutral = test_pred - np.mean(test_pred)
    
    # 2. Hard Clipping
    test_pred_clip = np.clip(test_pred_neutral, -0.02, 0.02)
    
    # 3. Shrinkage
    test_pred_clip = test_pred_clip * 0.9

    elapsed = (time.time() - t_h) / 60
    print(f'  Horizon {horizon} total: {elapsed:.1f} min')

    out = {
        'horizon': horizon,
        'id_test': te['id'].values,
        'test_pred_raw': test_pred,
        'test_pred_clip': test_pred_clip,
        'oof_y': oof_y[valid],
        'oof_w': oof_w[valid],
        'oof_pred': oof_pred,
        'score_local': score_ridge,
        'score_lgb': score_lgb,
        'score_cat': score_cat,
    }

    del tr, te, tr_all, te_feat, lfd, last_fold_data
    gc.collect()
    return out


def _train_lgb_fold_simple(X_tr, y_tr, w_tr, X_va, y_va, w_va, seeds):
    preds = np.zeros(len(X_va), dtype=np.float64)
    for seed in seeds:
        mdl = lgb.LGBMRegressor(**LGBM_BASE_PARAMS, n_estimators=5000, random_state=seed)
        mdl.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_va, y_va)], eval_sample_weight=[w_va],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        preds += mdl.predict(X_va) / len(seeds)
        del mdl
        gc.collect()
    return preds

def train_and_predict_all_horizons():
    print("=" * 70)
    print("LGB + CatBoost")
    print("=" * 70)
    print(f"CV: {N_CV_SPLITS}-fold TimeSeriesSplit")
    print(f"CV seeds: LGB={len(CV_LGB_SEEDS)}, CAT={len(CV_CAT_SEEDS)}")
    print(f"Final seeds: LGB={len(FINAL_LGB_SEEDS)}, CAT={len(FINAL_CAT_SEEDS)}")

    all_outputs = []
    for h in HORIZONS:
        all_outputs.append(solve_horizon(h))

    # ── Build submissions ──
    sub_clip_parts = []
    sub_raw_parts = []
    oof_parts = []

    for out in all_outputs:
        sub_clip_parts.append(pd.DataFrame({
            'id': out['id_test'], 'prediction': out['test_pred_clip'],
        }))
        sub_raw_parts.append(pd.DataFrame({
            'id': out['id_test'], 'prediction': out['test_pred_raw'],
        }))
        oof_parts.append(pd.DataFrame({
            'y_true': out['oof_y'], 'y_pred': out['oof_pred'],
            'w': out['oof_w'], 'horizon': out['horizon'],
        }))

    sub_clip = pd.concat(sub_clip_parts, ignore_index=True)
    sub_raw = pd.concat(sub_raw_parts, ignore_index=True)
    oof = pd.concat(oof_parts, ignore_index=True)

    # ── Aggregate score ──
    agg_score = weighted_rmse_score(oof['y_true'].values, oof['y_pred'].values, oof['w'].values)

    print()
    print('=' * 70)
    print('PER-HORIZON OOF SCORES (5-fold CV)')
    print(f'{"H":>4} | {"LGB":>10} | {"CAT":>10} | {"Ridge":>10}')
    print('-' * 42)
    for out in sorted(all_outputs, key=lambda d: d['horizon']):
        print(f"  {out['horizon']:>2} | {out['score_lgb']:.6f} | {out['score_cat']:.6f} | {out['score_local']:.6f}")
    print('-' * 42)
    print(f'  Aggregate OOF Ridge score: {agg_score:.6f}')
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
