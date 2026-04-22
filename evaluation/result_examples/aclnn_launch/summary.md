# Evaluation summary

- **Job ID**: `sim-job-1776828185-2179143`
- **Device**: NPU 3
- **Hardware**: 910b2
- **Final status**: SUCCEEDED
- **Operators**: 3
- **Cases passed**: 34/36
- **Overall geomean speedup**: 3.016x

## Stages

| Stage | Status | Duration |
| --- | --- | --- |
| prepare | passed | — |
| compile | passed | 0s |
| correctness | passed | 1m15s |
| performance | passed | 6m42s |
| archive | passed | — |

## Operators

| Operator | Passed | Geomean speedup |
| --- | --- | --- |
| Add | 8/8 | 4.476x |
| Sqrt | 8/8 | 5.775x |
| Mish | 18/20 | 1.061x |

### Add (8/8 passed, 4.476x)

| # | Case | Status | Speedup | Baseline µs | Custom µs | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `float32-1D-1K` | PASS | 10.821x | 17.53 | 1.62 | MERE=0.00e+00 MARE=0.00e+00 |
| 2 | `float32-2D-1M` | PASS | 1.671x | 18.25 | 10.92 | MERE=0.00e+00 MARE=0.00e+00 |
| 3 | `float16-2D-1M` | PASS | 3.159x | 19.65 | 6.22 | MERE=4.71e-05 MARE=4.88e-04 |
| 4 | `float32-2D-对称中值域` | PASS | 5.309x | 18.58 | 3.50 | MERE=0.00e+00 MARE=0.00e+00 |
| 5 | `float16-4D-图像batch` | PASS | 7.992x | 18.94 | 2.37 | MERE=9.16e-05 MARE=4.88e-04 |
| 6 | `int32-1D-1K` | PASS | 12.709x | 19.19 | 1.51 | exact match |
| 7 | `float32-非对齐-2D` | PASS | 1.724x | 19.15 | 11.11 | MERE=0.00e+00 MARE=0.00e+00 |
| 8 | `float16-1D-质数非对齐` | PASS | 3.036x | 19.55 | 6.44 | MERE=4.73e-05 MARE=4.88e-04 |

### Sqrt (8/8 passed, 5.775x)

| # | Case | Status | Speedup | Baseline µs | Custom µs | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `float32-1D-1K` | PASS | 14.074x | 17.17 | 1.22 | MERE=6.62e-09 MARE=1.19e-07 |
| 2 | `float32-2D-1M` | PASS | 2.601x | 18.70 | 7.19 | MERE=6.69e-09 MARE=1.19e-07 |
| 3 | `float16-2D-1M` | PASS | 4.032x | 19.07 | 4.73 | MERE=1.80e-04 MARE=4.88e-04 |
| 4 | `float32-2D-对称小值域` | PASS | 6.383x | 18.51 | 2.90 | MERE=6.62e-09 MARE=1.19e-07 |
| 5 | `float16-4D-图像batch` | PASS | 6.819x | 19.16 | 2.81 | MERE=1.59e-04 MARE=4.60e-04 |
| 6 | `float32-1D-零值` | PASS | 15.598x | 19.03 | 1.22 | MERE=0.00e+00 MARE=0.00e+00 |
| 7 | `float32-非对齐-2D` | PASS | 2.762x | 18.89 | 6.84 | MERE=6.71e-09 MARE=1.19e-07 |
| 8 | `float16-1D-质数非对齐` | PASS | 4.471x | 18.24 | 4.08 | MERE=1.80e-04 MARE=4.88e-04 |

### Mish (18/20 passed, 1.061x)

| # | Case | Status | Speedup | Baseline µs | Custom µs | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `float16-1M-对齐-对称小值域` | PASS | 2.960x | 40.20 | 13.58 | MERE=1.78e-04 MARE=4.80e-04 |
| 2 | `float32-4M-对齐-对称小值域` | PASS | 3.122x | 132.87 | 42.56 | MERE=3.69e-08 MARE=2.38e-07 |
| 3 | `bfloat16-16M-对齐-对称小值域` | PASS | 0.242x | 41.53 | 171.49 | MERE=1.36e-03 MARE=3.42e-03 |
| 4 | `float16-67M-对齐-对称中值域-2D扁平` | PASS | 0.060x | 40.24 | 672.35 | MERE=1.41e-04 MARE=4.70e-04 |
| 5 | `float32-268M-对齐-对称大值域-2D` | PASS | 0.015x | 38.95 | 2579.44 | MERE=1.47e-07 MARE=7.00e-06 |
| 6 | `bfloat16-1M-非对齐-对称微小值域` | PASS | 3.035x | 41.12 | 13.55 | MERE=1.40e-03 MARE=3.71e-03 |
| 7 | `float16-1M-质数非对齐-非对称小值域-5D` | PASS | 26.382x | 354.04 | 13.42 | MERE=1.74e-04 MARE=4.80e-04 |
| 8 | `float32-1M-非对齐-非对称中值域` | PASS | 2.972x | 43.12 | 14.51 | MERE=3.79e-08 MARE=2.64e-07 |
| 9 | `bfloat16-50M-质数非对齐-非对称大值域-3D` | PASS | 3.950x | 1979.58 | 501.19 | MERE=4.94e-04 MARE=3.83e-03 |
| 10 | `float16-1M-非对齐-float16边界值` | FAIL | — | 128.34 | — | MERE=5.41e-04(limit 9.77e-04) MARE=1.00e+00(limit 9.77e-03) |
| 11 | `float32-1M-质数非对齐-float32边界值-4D` | PASS | 2.996x | 40.12 | 13.39 | MERE=1.58e-08 MARE=2.52e-07 |
| 12 | `bfloat16-1M-质数非对齐-inf特殊值-1D` | PASS | 3.118x | 41.53 | 13.32 | MERE=0.00e+00 MARE=0.00e+00 |
| 13 | `float32-10M-质数非对齐-nan特殊值-5D` | PASS | 2.964x | 314.17 | 106.01 | all special values matched |
| 14 | `float16-3M-质数非对齐-零值-5D` | PASS | 2.161x | 71.65 | 33.16 | MERE=0.00e+00 MARE=0.00e+00 |
| 15 | `float32-1M-非对齐-对称微小值域` | PASS | 3.040x | 39.22 | 12.90 | MERE=3.78e-08 MARE=2.38e-07 |
| 16 | `bfloat16-2M-非对齐-非对称小值域` | PASS | 0.289x | 6.88 | 23.78 | MERE=1.34e-03 MARE=3.39e-03 |
| 17 | `float16-2M-非对齐-对称大值域` | FAIL | — | 40.20 | — | MERE=3.60e-02(limit 9.77e-04) MARE=1.00e+00(limit 9.77e-03) |
| 18 | `float32-2M-非对齐-对称微小值域-3D` | PASS | 1.717x | 38.93 | 22.67 | MERE=4.07e-08 MARE=2.39e-07 |
| 19 | `bfloat16-2M-非对齐-非对称中值域-3D` | PASS | 0.315x | 7.54 | 23.91 | MERE=1.10e-03 MARE=3.62e-03 |
| 20 | `float32-10M-非对齐-非对称大值域-5D` | PASS | 0.054x | 5.49 | 101.80 | MERE=1.98e-08 MARE=3.31e-07 |
