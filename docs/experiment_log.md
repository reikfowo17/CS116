# Experiment Log

## Benchmark Scores

| # | Date | Model | Features | Val Score | Public LB | Notes |
|---|------|-------|----------|-----------|-----------|-------|
| 1 | 16/03 | LightGBM v1 (baseline) | Raw + 3 lag (wrong groupby) | 0.10803 | — | Early stop ở round 10 |
| 2 | 18/03 | LightGBM v2 (fixed pipeline) | Raw + lag(1,2,3) + rolling(3,5) + diff | H1:0.031, H3:0.052, H10:0.112, H25:0.141 | — | lr=0.05, num_leaves=63 |
| 3 | 18/03 | LightGBM v3 (optimized) | + EWM + ratio + pctchg | H1:0.037, H3:0.058, H10:0.075, H25:0.157 | 0.1838 | lr=0.01, num_leaves=127 |
| 4 | 19/03 | LightGBM v4 (20-seed + retrain-on-all) | Interactions + TargetEnc + CS-norm + Cyclical + Lag/Roll/EWM/Diff/Rank (~170 feats) | H1:0.080, H3:0.140, H10:0.222, H25:0.272 (Agg: 0.2353) | 0.2612 | 20 seeds, retrain-on-all, clipping, L1/L2 reg |
| 5 | 20/03 | LightGBM v4.1 (+ feature selection) | Top 60 features (importance-based) | — | — | Probe model → feature importance → keep top 60 |

## Notes
- Score range: 0 (worst) → 1 (best)
- Public LB uses 25% of test data
- Private LB uses remaining 75%
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
  - **Feature Selection**: Train 1 probe model → lấy feature_importance → giữ top 60/~170 features
  - Loại bỏ features noise → model tập trung vào signal mạnh
  - In ra top 15 features per horizon để phân tích
