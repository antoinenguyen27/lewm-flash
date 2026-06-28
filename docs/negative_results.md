# Negative Results

LeWM-Flash keeps rejected paths visible because they matter for interpreting the
final result.

| Branch | Outcome |
|---|---|
| Receding-horizon CEM warm start | Rejected. Shifted previous CEM mean/variance collapsed Push-T quality. |
| Action-tail-only warm start | Rejected. Removing posterior reuse did not fix quality collapse. |
| Rollout preallocation | Rejected. Intended allocation savings were outweighed by slower predictor/rollout execution. |
| Action-embedding cache | Diagnostic only. Fixed-candidate exactness passed, but broader timing/quality did not promote. |
| Whole-chain static iCEM `torch.compile` | Rejected. Compilation finished but generated thousands of kernels and had unacceptable warmup/first-plan behavior. |
| CEM-GD/iCEM-GD first path | Diagnostic only. Lowered model cost but did not improve task quality and regressed wall time. |
| Static eager BF16 | Rejected. Passed quality but regressed aggregate planning and eval wall versus accepted iCEM + BF16. |

See `docs/research/` for the detailed experiment reports.
