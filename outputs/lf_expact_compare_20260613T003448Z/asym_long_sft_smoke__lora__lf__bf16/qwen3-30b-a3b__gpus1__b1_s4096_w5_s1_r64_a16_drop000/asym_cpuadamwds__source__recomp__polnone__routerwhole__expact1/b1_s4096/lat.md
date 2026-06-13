# LF LoRA-SFT Latency

| Metric | ms |
|---|---:|
| trainer e2e measured step incl optimizer | 16020.641 |
| trainer e2e total step incl warmup | 55900.132 |
| optimizer/update side = e2e measured - fwd/bwd | 1401.636 |
| lf.training_step.total | 14660.179 |
| step.forward + step.backward | 14619.005 |
| step.forward | 600.945 |
| step.backward | 14018.059 |
| lf.grad_clip | 5.600 |
| lf.optimizer.step substage | 994.396 |
| lf.scheduler.step | 0.279 |

Timing source: `heartbeat_dataloader_interval`
