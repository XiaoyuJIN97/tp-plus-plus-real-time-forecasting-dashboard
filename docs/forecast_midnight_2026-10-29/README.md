# Forecast window migration — 29 October 2026

Forecasts now cover 18:00 Europe/Brussels through midnight at the end of the following day (exclusive): normally 30 hours, 29/31 across daylight saving changes. The last hourly timestamp is 23:00 local and represents the hour ending at midnight.

Historical replay covers origins 24 July–28 October: 17,465 complete model windows and 2,495 ensemble windows. The 868 incomplete model windows remain excluded from comparisons; no missing observations were replaced with invented truth. Original shorter records remain archived. Reconstructed forecasts use archived/recovered inputs, not certified issue-time vintages. Input hashes and complete/excluded inventories are attached.

Point and probabilistic leaderboards were rebuilt on shared, complete observed windows. Dashboard retention remains 90 days. The full 2024/2025 offline plan and pipeline were updated; the entire two-year benchmark was not rerun in this migration.

Validation: 133 regression tests passed; native adapters exercised 29/30/31-hour windows. Batch/single CPU differences were below 0.002 MW; historical T0 Gamma/Timestral MPS acceleration agreed with CPU within 0.005 MW. Production inference remains CPU. Spring 29-hour integration used synthetic inputs only, excluded from published history.
