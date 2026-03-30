# Experiment Log

## Benchmark Scores

| # | Date | Model | Features | Val Score | Public LB | Notes |
|---|------|-------|----------|-----------|-----------|-------|
| 1 | 16/03 | LightGBM v1 (baseline) | Raw + 3 lag (wrong groupby) | 0.10803 | — | Early stop ở round 10 |
| 2 | 18/03 | LightGBM v2 (fixed pipeline) | Raw + lag(1,2,3) + rolling(3,5) + diff | H1:0.031, H3:0.052, H10:0.112, H25:0.141 | — | lr=0.05, num_leaves=63 |
| 3 | 18/03 | LightGBM v3 (optimized) | + EWM + ratio + pctchg | H1:0.037, H3:0.058, H10:0.075, H25:0.157 | 0.1838 | lr=0.01, num_leaves=127 |
| 4 | 19/03 | LightGBM v4 (20-seed + retrain-on-all) | Interactions + TargetEnc + CS-norm + Cyclical + Lag/Roll/EWM/Diff/Rank (~170 feats) | H1:0.080, H3:0.140, H10:0.222, H25:0.272 (Agg: 0.2353) | 0.2612 | 20 seeds, retrain-on-all, clipping, L1/L2 reg |
| 5 | 20/03 | LightGBM v4.1 (+ feature selection) | Top 50-65 features per-horizon (importance-based) | H1:0.078, H3:0.139, H10:0.219, H25:0.275 (Agg: 0.2357) | 0.2566 ❌ | Feature selection giảm score so với v4 |
| 6 | 25/03 | LGB+CAT v5 (Ridge stacking, 5-fold CV) | V4 feats + time_phase + lifecycle + momentum + roll_min/max + target enc `code`, percentile clip (~150+ feats) | H1:0.047, H3:0.057, H10:0.107, H25:0.135 (Agg: 0.116) | — | Ridge stacking LGB+CAT, skip CAT H=10,25, percentile clip [p0.5,p99.5] |
| 7 | 26/03 | LGB v6 (Hybrid — Calibration + Concat + FreqEnc) | 190 feats, 15 LGB seeds, no CAT, linear calibration, train+test concat, freq encoding | H1:0.064, H3:0.110, H10:0.226, H25:0.297 (Agg: 0.248) | 0.2462 | Pure LGB 15-seed, ~4h runtime, details below |
| 8 | 30/03 | LGB v7 (Enhanced v4) — Pseudo-targets + Target Lags + Recency Weighting + Native Categoricals | ~170 feats, 15 seeds, per-horizon scale grid, feature pruning top-100, LGB native API + skill metric | H1:0.095, H3:0.175, H10:0.291, H25:0.355 (Agg: 0.298) | — | Merged v4 enhancements into src/ module; pseudo-target features (feature_al/am/cg/s shifted by optimal shifts per horizon); target lag features (y_lag1, y_lag3, y_diff1, y_expand_mean); linear recency weighting (factor=42); LGB native API with categorical_feature; per-horizon scale grid search; feature pruning top-100 by gain |

## Notes
- Score range: 0 (worst) → 1 (best)
- Public LB uses 25% of test data
- Private LB uses remaining 75%
- **Host confirmed**: score > 0.5 = data leak, Private LB ≠ Public LB. "Clean" max ~0.35-0.40.
- **Notebook prize requirement**: must run < 6 hours
- v2 → v3 changes: giảm lr 0.05→0.01, tăng num_leaves 63→127, thêm EWM/ratio/pct_change features
- v3 → v4 changes (học hỏi từ notebook top scorer 0.2359):
  - **20-seed ensemble**: Train 20 models ngẫu nhiên thay vì chỉ 1 hoặc 3
  - **Retrain-on-All**: Tìm best_iteration → train lại trên TOÀN BỘ data (train+val) → predict test
  - **Target Encoding**: mean(y_target) theo sub_category và sub_code
  - **Feature Interactions**: al-am, al/am, cg-by, s*al, s*am, s*cg
  - **Rolling std**: Thêm rolling std bên cạnh rolling mean
  - **Feature rank**: rank(pct=True) theo ts_index
  - **Cyclical time**: sin/cos(2π·ts/100)
  - **Regularization**: L1=0.1, L2=10.0
  - **Clipping**: Clip predictions theo quantile [0.005, 0.995]
  - **Hyperparams**: lr=0.015, num_leaves=90, min_child_samples=200
  - **Val split**: ts_index=3500
- v4 → v4.1 changes:
  - **Feature Selection**: Train 1 probe model → lấy feature_importance → giữ top 50-65 features
  - ❌ Kết quả: giảm score → đã revert (disable feature selection)
- v4 → v5 changes:
  - **CatBoost**: Thêm CatBoost (ordered boosting, categoricals-friendly) thay XGBoost
  - **Ridge Stacking**: Ridge(alpha=1.0) học optimal weights LGB+CAT từ OOF
  - **5-Fold TimeSeriesSplit**: Thay validation 1 fold cuối bằng 5 nếp gấp thời gian
  - **Per-Fold Target Encoding**: Triệt tiêu target leakage
  - **New Features**: time_phase, lifecycle, momentum, roll_min/max
  - **Anti-leakage**: shift(1) trước rolling trên target-derived features
  - **LGBM Params**: num_leaves 90→127, L1 0.1→2.0, extra_trees, path_smooth
  - **Skip CatBoost** cho H=10, H=25 (score = 0 → vô ích nhưng tốn ~4h)
  - **Thêm target encoding `code`**: Mỗi code tài sản có hành vi riêng
  - **Post-Processing**: Percentile clip [p0.5, p99.5]
- v5 → v6 changes:
  - **Loại bỏ CatBoost hoàn toàn**: Pure LGB, dồn toàn bộ budget vào 15 seeds
  - **Calibration**: Linear `a×pred + b` trên OOF (counter GBDT shrinkage)
  - **Train+Test concat**: Concat trước khi build lags → test rows có lag từ cuối train
  - **Frequency Encoding**: Thêm freq encoding cho code/sub_code/sub_category (rule-safe, no target)
  - **Hold-out validation**: Thay 5-fold CV bằng simple split ts_index=3500 → nhanh hơn, dồn time cho retrain
  - **15 LGB seeds**: Scale up từ 7 seeds → ổn định hơn
  - **Estimated runtime**: ~4h (< 6h limit)
  - **Kaggle Discussion insight**: Host xác nhận score > 0.5 là dùng data leak. Private LB sẽ neutralize. Sequential prediction bắt buộc.
- v6 → v7 changes (Enhanced v4):
  - **Pseudo-target features**: `feature_al/am/cg/s` shifted by `OPTIMAL_SHIFTS[horizon]` = giá trị từ h steps trước → causal signal mạnh nhất
  - **Target lag features**: `y_lag1`, `y_lag3`, `y_diff1`, `y_expand_mean` (với shift(1) để tránh leakage)
  - **Recency weighting**: Linear up-weighting recent observations với `RECENCY_FACTOR=42.0`
  - **Native categorical handling**: Dùng LightGBM `categorical_feature` parameter
  - **Per-horizon scale grid search**: Grid search scale factor tối ưu cho từng horizon
  - **Feature pruning**: Giữ top-100 features theo gain importance
  - **LGB Native API + Skill Metric**: Dùng `lgb.Dataset`, `lgb.train` với custom `feval=lgb_skill_metric`
  - **Merged into src/ module**: Tất cả enhancements được apply vào `src/config.py`, `src/features.py`, `src/models.py`
  - **15 seeds**: Giữ nguyên từ v6
