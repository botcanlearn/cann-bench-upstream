# Evaluation summary

- **Job ID**: `sim-job-1776827847-2164471`
- **Device**: NPU 3
- **Hardware**: 910b2
- **Final status**: SUCCEEDED
- **Operators**: 3
- **Cases passed**: 30/36
- **Overall geomean speedup**: 3.398x

## Stages

| Stage | Status | Duration |
| --- | --- | --- |
| prepare | passed | — |
| compile | passed | 0s |
| correctness | passed | 47s |
| performance | passed | 4m27s |
| archive | passed | — |

## Operators

| Operator | Passed | Geomean speedup |
| --- | --- | --- |
| Add | 8/8 | 4.139x |
| Sqrt | 8/8 | 3.712x |
| Mish | 14/20 | 2.554x |

### Add (8/8 passed, 4.139x)

| # | Case | Status | Speedup | Baseline µs | Custom µs | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `float32-1D-1K` | PASS | 8.510x | 17.53 | 2.06 | MERE=0.00e+00 MARE=0.00e+00 |
| 2 | `float32-2D-1M` | PASS | 1.800x | 18.25 | 10.14 | MERE=0.00e+00 MARE=0.00e+00 |
| 3 | `float16-2D-1M` | PASS | 2.995x | 19.65 | 6.56 | MERE=4.71e-05 MARE=4.88e-04 |
| 4 | `float32-2D-对称中值域` | PASS | 5.370x | 18.58 | 3.46 | MERE=0.00e+00 MARE=0.00e+00 |
| 5 | `float16-4D-图像batch` | PASS | 6.149x | 18.94 | 3.08 | MERE=9.16e-05 MARE=4.88e-04 |
| 6 | `int32-1D-1K` | PASS | 8.843x | 19.19 | 2.17 | exact match |
| 7 | `float32-非对齐-2D` | PASS | 1.927x | 19.15 | 9.94 | MERE=0.00e+00 MARE=0.00e+00 |
| 8 | `float16-1D-质数非对齐` | PASS | 3.336x | 19.55 | 5.86 | MERE=4.73e-05 MARE=4.88e-04 |

### Sqrt (8/8 passed, 3.712x)

| # | Case | Status | Speedup | Baseline µs | Custom µs | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `float32-1D-1K` | PASS | 6.046x | 17.17 | 2.84 | MERE=6.62e-09 MARE=1.19e-07 |
| 2 | `float32-2D-1M` | PASS | 2.320x | 18.70 | 8.06 | MERE=6.69e-09 MARE=1.19e-07 |
| 3 | `float16-2D-1M` | PASS | 2.526x | 19.07 | 7.55 | MERE=1.80e-04 MARE=4.88e-04 |
| 4 | `float32-2D-对称小值域` | PASS | 3.479x | 18.51 | 5.32 | MERE=6.62e-09 MARE=1.19e-07 |
| 5 | `float16-4D-图像batch` | PASS | 4.230x | 19.16 | 4.53 | MERE=1.59e-04 MARE=4.60e-04 |
| 6 | `float32-1D-零值` | PASS | 12.438x | 19.03 | 1.53 | MERE=0.00e+00 MARE=0.00e+00 |
| 7 | `float32-非对齐-2D` | PASS | 2.284x | 18.89 | 8.27 | MERE=6.71e-09 MARE=1.19e-07 |
| 8 | `float16-1D-质数非对齐` | PASS | 2.432x | 18.24 | 7.50 | MERE=1.80e-04 MARE=4.88e-04 |

### Mish (14/20 passed, 2.554x)

| # | Case | Status | Speedup | Baseline µs | Custom µs | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `float16-1M-对齐-对称小值域` | PASS | 4.258x | 40.20 | 9.44 | MERE=1.78e-04 MARE=4.80e-04 |
| 2 | `float32-4M-对齐-对称小值域` | PASS | 6.270x | 132.87 | 21.19 | MERE=6.32e-08 MARE=8.26e-07 |
| 3 | `bfloat16-16M-对齐-对称小值域` | PASS | 0.545x | 41.53 | 76.17 | MERE=1.36e-03 MARE=3.42e-03 |
| 4 | `float16-67M-对齐-对称中值域-2D扁平` | PASS | 0.142x | 40.24 | 283.77 | MERE=1.56e-04 MARE=1.23e-03 |
| 5 | `float32-268M-对齐-对称大值域-2D` | FAIL | — | 38.95 | — | MERE=3.80e-01(limit 1.22e-04) MARE=1.00e+00(limit 1.22e-03) |
| 6 | `bfloat16-1M-非对齐-对称微小值域` | PASS | 4.315x | 41.12 | 9.53 | MERE=1.40e-03 MARE=3.71e-03 |
| 7 | `float16-1M-质数非对齐-非对称小值域-5D` | PASS | 36.995x | 354.04 | 9.57 | MERE=1.74e-04 MARE=4.80e-04 |
| 8 | `float32-1M-非对齐-非对称中值域` | PASS | 4.158x | 43.12 | 10.37 | MERE=3.30e-07 MARE=1.39e-05 |
| 9 | `bfloat16-50M-质数非对齐-非对称大值域-3D` | FAIL | — | 1979.58 | — | MERE=2.28e-01(limit 7.81e-03) MARE=1.00e+00(limit 7.81e-02) |
| 10 | `float16-1M-非对齐-float16边界值` | FAIL | — | 128.34 | — | MERE=5.84e-04(limit 9.77e-04) MARE=1.00e+00(limit 9.77e-03) |
| 11 | `float32-1M-质数非对齐-float32边界值-4D` | FAIL | — | 40.12 | — | MERE=4.08e-01(limit 1.22e-04) MARE=1.00e+00(limit 1.22e-03) |
| 12 | `bfloat16-1M-质数非对齐-inf特殊值-1D` | PASS | 4.358x | 41.53 | 9.53 | MERE=0.00e+00 MARE=0.00e+00 |
| 13 | `float32-10M-质数非对齐-nan特殊值-5D` | PASS | 6.722x | 314.17 | 46.74 | all special values matched |
| 14 | `float16-3M-质数非对齐-零值-5D` | PASS | 4.023x | 71.65 | 17.81 | MERE=0.00e+00 MARE=0.00e+00 |
| 15 | `float32-1M-非对齐-对称微小值域` | PASS | 4.245x | 39.22 | 9.24 | MERE=4.42e-08 MARE=3.24e-07 |
| 16 | `bfloat16-2M-非对齐-非对称小值域` | PASS | 0.493x | 6.88 | 13.96 | MERE=1.34e-03 MARE=3.39e-03 |
| 17 | `float16-2M-非对齐-对称大值域` | FAIL | — | 40.20 | — | MERE=3.80e-02(limit 9.77e-04) MARE=1.00e+00(limit 9.77e-03) |
| 18 | `float32-2M-非对齐-对称微小值域-3D` | PASS | 2.763x | 38.93 | 14.09 | MERE=4.72e-08 MARE=2.97e-07 |
| 19 | `bfloat16-2M-非对齐-非对称中值域-3D` | PASS | 0.539x | 7.54 | 13.98 | MERE=1.10e-03 MARE=3.62e-03 |
| 20 | `float32-10M-非对齐-非对称大值域-5D` | FAIL | — | 5.49 | — | MERE=6.52e-02(limit 1.22e-04) MARE=1.00e+00(limit 1.22e-03) |
