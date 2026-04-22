# Evaluation summary

- **Job ID**: `sim-job-1776837446-2371027`
- **Device**: NPU 1
- **Hardware**: 910b2
- **Final status**: SUCCEEDED
- **Operators**: 2
- **Cases passed**: 16/16
- **Overall geomean speedup**: 3.884x

## Stages

| Stage | Status | Duration |
| --- | --- | --- |
| prepare | passed | — |
| compile | passed | 0s |
| correctness | passed | 16s |
| performance | passed | 2m17s |
| archive | passed | — |

## Operators

| Operator | Passed | Geomean speedup |
| --- | --- | --- |
| Add | 8/8 | 4.138x |
| Sqrt | 8/8 | 3.647x |

### Add (8/8 passed, 4.138x)

| # | Case | Status | Speedup | Baseline µs | Custom µs | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `float32-1D-1K` | PASS | 8.678x | 17.53 | 2.02 | MERE=0.00e+00 MARE=0.00e+00 |
| 2 | `float32-2D-1M` | PASS | 1.870x | 18.25 | 9.76 | MERE=0.00e+00 MARE=0.00e+00 |
| 3 | `float16-2D-1M` | PASS | 3.253x | 19.65 | 6.04 | MERE=4.71e-05 MARE=4.88e-04 |
| 4 | `float32-2D-对称中值域` | PASS | 5.161x | 18.58 | 3.60 | MERE=0.00e+00 MARE=0.00e+00 |
| 5 | `float16-4D-图像batch` | PASS | 6.230x | 18.94 | 3.04 | MERE=9.16e-05 MARE=4.88e-04 |
| 6 | `int32-1D-1K` | PASS | 8.567x | 19.19 | 2.24 | exact match |
| 7 | `float32-非对齐-2D` | PASS | 1.946x | 19.15 | 9.84 | MERE=0.00e+00 MARE=0.00e+00 |
| 8 | `float16-1D-质数非对齐` | PASS | 3.036x | 19.55 | 6.44 | MERE=4.73e-05 MARE=4.88e-04 |

### Sqrt (8/8 passed, 3.647x)

| # | Case | Status | Speedup | Baseline µs | Custom µs | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `float32-1D-1K` | PASS | 6.154x | 17.17 | 2.79 | MERE=6.62e-09 MARE=1.19e-07 |
| 2 | `float32-2D-1M` | PASS | 2.237x | 18.70 | 8.36 | MERE=6.69e-09 MARE=1.19e-07 |
| 3 | `float16-2D-1M` | PASS | 2.581x | 19.07 | 7.39 | MERE=1.80e-04 MARE=4.88e-04 |
| 4 | `float32-2D-对称小值域` | PASS | 3.305x | 18.51 | 5.60 | MERE=6.62e-09 MARE=1.19e-07 |
| 5 | `float16-4D-图像batch` | PASS | 4.202x | 19.16 | 4.56 | MERE=1.59e-04 MARE=4.60e-04 |
| 6 | `float32-1D-零值` | PASS | 11.464x | 19.03 | 1.66 | MERE=0.00e+00 MARE=0.00e+00 |
| 7 | `float32-非对齐-2D` | PASS | 2.306x | 18.89 | 8.19 | MERE=6.71e-09 MARE=1.19e-07 |
| 8 | `float16-1D-质数非对齐` | PASS | 2.397x | 18.24 | 7.61 | MERE=1.80e-04 MARE=4.88e-04 |
