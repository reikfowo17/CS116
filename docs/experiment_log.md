# Experiment Log

## Benchmark Scores

| # | Date | Model | Features | Val Score | Public LB | Notes |
|---|------|-------|----------|-----------|-----------|-------|
| 1 | 16/03 | LightGBM v1 (baseline) | Raw + 3 lag (wrong groupby) | 0.10803 | — | Early stop ở round 10 |
| 2 | 18/03 | LightGBM v2 (fixed pipeline) | Raw + lag(1,2,3) + rolling(3,5) + diff | H1:0.031, H3:0.052, H10:0.112, H25:0.141 | — | lr=0.05, num_leaves=63, early stop 10-48 rounds |
| 3 | | LightGBM v3 (optimized) | Raw + lag(1,2,3,5,7,10) + rolling(3,5,10,20) + EWM + ratio + pctchg | — | — | lr=0.01, num_leaves=127, chờ chạy |

## Notes
- Score range: 0 (worst) → 1 (best)
- Public LB uses 25% of test data
- Private LB uses remaining 75%
- v2 → v3 changes: giảm lr 0.05→0.01, tăng num_leaves 63→127, thêm EWM/ratio/pct_change features
