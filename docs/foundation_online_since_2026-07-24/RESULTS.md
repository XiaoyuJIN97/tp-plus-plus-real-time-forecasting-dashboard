# T0 Gamma and Timestral online mock replay

Mock forecasts follow each saved TimesFM3 delivery window from 24 July through 19 October 2026, with 2,208 context hours and 24 forecast hours. Actual context ends one hour before delivery. The model checkpoints and inference settings match the validated offline evaluation.

All rows are labeled `historical_mock_replay`. Collector archives and retained outcomes reconstruct past inputs. October weather gaps use the Open-Meteo historical forecast API; these are not certified original issue-time vintages. This is a reconstructed mock comparison, not a claim of contemporaneous live performance.

Coverage gaps remain excluded under production context-quality limits. Some older baseline horizons differ by two hours; all comparisons below match the exact run date and delivery timestamp, using the same realized actuals for both models. Pairs with different forecast origins are marked in the comparison CSV and are descriptive rather than controlled comparisons.

Saved predictions: **108,048 rows / 4,502 complete daily windows** across 7 zones.

## Matched comparison with TimesFM3

| Zone | Target | Model | Scored hours | MAE (MW) | TimesFM3 MAE | Change |
|---|---|---|---:|---:|---:|---:|
| BE | load | t0_gamma | 2103 | 214.25 | 216.38 | -1.0% |
| BE | load | timestral | 2103 | 214.86 | 216.38 | -0.7% |
| BE | solar | t0_gamma | 2100 | 188.51 | 182.15 | +3.5% |
| BE | solar | timestral | 2100 | 174.76 | 182.15 | -4.1% |
| BE | wind_offshore | t0_gamma | 2037 | 132.36 | 120.30 | +10.0% |
| BE | wind_offshore | timestral | 2037 | 128.69 | 120.30 | +7.0% |
| BE | wind_onshore | t0_gamma | 2100 | 96.64 | 87.37 | +10.6% |
| BE | wind_onshore | timestral | 2100 | 94.24 | 87.37 | +7.9% |
| DE | load | t0_gamma | 2103 | 1227.92 | 1177.96 | +4.2% |
| DE | load | timestral | 2103 | 1148.83 | 1177.96 | -2.5% |
| DE | solar | t0_gamma | 2103 | 946.37 | 827.70 | +14.3% |
| DE | solar | timestral | 2103 | 768.03 | 827.70 | -7.2% |
| DE | wind_offshore | t0_gamma | 2103 | 540.20 | 507.89 | +6.4% |
| DE | wind_offshore | timestral | 2103 | 519.47 | 507.89 | +2.3% |
| DE | wind_onshore | t0_gamma | 2103 | 1108.17 | 1057.17 | +4.8% |
| DE | wind_onshore | timestral | 2103 | 1050.18 | 1057.17 | -0.7% |
| DK1 | load | t0_gamma | 2012 | 52.88 | 46.07 | +14.8% |
| DK1 | load | timestral | 2012 | 43.00 | 46.07 | -6.7% |
| DK1 | solar | t0_gamma | 961 | 86.20 | 74.74 | +15.3% |
| DK1 | solar | timestral | 961 | 79.85 | 74.74 | +6.8% |
| DK1 | wind_offshore | t0_gamma | 2056 | 107.25 | 108.02 | -0.7% |
| DK1 | wind_offshore | timestral | 2056 | 103.84 | 108.02 | -3.9% |
| DK1 | wind_onshore | t0_gamma | 2104 | 127.68 | 126.23 | +1.1% |
| DK1 | wind_onshore | timestral | 2104 | 124.48 | 126.23 | -1.4% |
| DK2 | load | t0_gamma | 2012 | 30.13 | 27.06 | +11.3% |
| DK2 | load | timestral | 2012 | 25.19 | 27.06 | -6.9% |
| DK2 | solar | t0_gamma | 712 | 34.31 | 31.18 | +10.0% |
| DK2 | solar | timestral | 712 | 31.92 | 31.18 | +2.4% |
| DK2 | wind_offshore | t0_gamma | 2056 | 69.53 | 74.18 | -6.3% |
| DK2 | wind_offshore | timestral | 2056 | 76.91 | 74.18 | +3.7% |
| DK2 | wind_onshore | t0_gamma | 2104 | 21.29 | 20.15 | +5.7% |
| DK2 | wind_onshore | timestral | 2104 | 21.11 | 20.15 | +4.8% |
| ES | load | t0_gamma | 2104 | 423.30 | 335.51 | +26.2% |
| ES | load | timestral | 2104 | 323.52 | 335.51 | -3.6% |
| ES | solar | t0_gamma | 2056 | 678.96 | 688.75 | -1.4% |
| ES | solar | timestral | 2056 | 644.89 | 688.75 | -6.4% |
| ES | wind_onshore | t0_gamma | 2056 | 472.79 | 440.10 | +7.4% |
| ES | wind_onshore | timestral | 2056 | 445.48 | 440.10 | +1.2% |
| FR | load | t0_gamma | 2102 | 607.84 | 577.20 | +5.3% |
| FR | load | timestral | 2102 | 552.80 | 577.20 | -4.2% |
| FR | solar | t0_gamma | 2078 | 387.12 | 364.92 | +6.1% |
| FR | solar | timestral | 2078 | 341.93 | 364.92 | -6.3% |
| FR | wind_offshore | t0_gamma | 1967 | 111.89 | 105.33 | +6.2% |
| FR | wind_offshore | timestral | 1967 | 106.31 | 105.33 | +0.9% |
| FR | wind_onshore | t0_gamma | 2078 | 418.36 | 397.34 | +5.3% |
| FR | wind_onshore | timestral | 2078 | 402.08 | 397.34 | +1.2% |
| PT | load | t0_gamma | 2103 | 101.07 | 90.96 | +11.1% |
| PT | load | timestral | 2103 | 85.45 | 90.96 | -6.1% |
| PT | solar | t0_gamma | 2103 | 94.31 | 91.30 | +3.3% |
| PT | solar | timestral | 2103 | 84.72 | 91.30 | -7.2% |
| PT | wind_offshore | t0_gamma | 2103 | 2.16 | 2.03 | +6.5% |
| PT | wind_offshore | timestral | 2103 | 2.21 | 2.03 | +9.2% |
| PT | wind_onshore | t0_gamma | 2103 | 151.49 | 156.05 | -2.9% |
| PT | wind_onshore | timestral | 2103 | 153.75 | 156.05 | -1.5% |

See `coverage.csv` for individual model coverage, `replay_issues.csv` for excluded windows, and `matched_comparison.csv` for comparisons with the other saved models. Negative changes indicate lower MAE.

## Publication validation

Both native models passed real Linux CPU inference on Python 3.11; 55 production tests pass locally. Eight real-data CPU/GPU comparisons had a maximum difference of 0.00245 MW.

The import preserves all 359688 existing forecast predictions within CSV round-trip precision. It refreshed 7216 realized observations from the common collector snapshot so every model is scored against the same actuals. See `actual_refresh_audit.csv` and `import_validation.json`.
