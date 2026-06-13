# LF LoRA-SFT Latency

| Metric | ms |
|---|---:|
| trainer e2e measured step incl optimizer | 14708.241 |
| trainer e2e total step incl warmup | 12068.702 |
| optimizer/update side = e2e measured - fwd/bwd | 1423.797 |
| lf.training_step.total | 13365.610 |
| step.forward + step.backward | 13284.444 |
| step.forward | 5395.460 |
| step.backward | 7888.983 |
| lf.grad_clip | 5.757 |
| lf.optimizer.step substage | 996.247 |
| lf.scheduler.step | 0.267 |

Timing source: `heartbeat_dataloader_interval`
