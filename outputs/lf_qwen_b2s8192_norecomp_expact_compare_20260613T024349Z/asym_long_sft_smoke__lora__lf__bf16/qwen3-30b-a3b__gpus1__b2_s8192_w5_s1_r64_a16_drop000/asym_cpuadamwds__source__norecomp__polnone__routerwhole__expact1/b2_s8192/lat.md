# LF LoRA-SFT Latency

| Metric | ms |
|---|---:|
| trainer e2e measured step incl optimizer | 44949.827 |
| trainer e2e total step incl warmup | 78187.963 |
| optimizer/update side = e2e measured - fwd/bwd | 1432.244 |
| lf.training_step.total | 43599.920 |
| step.forward + step.backward | 43517.583 |
| step.forward | 10578.164 |
| step.backward | 32939.419 |
| lf.grad_clip | 6.535 |
| lf.optimizer.step substage | 1012.923 |
| lf.scheduler.step | 0.414 |

Timing source: `heartbeat_dataloader_interval`
