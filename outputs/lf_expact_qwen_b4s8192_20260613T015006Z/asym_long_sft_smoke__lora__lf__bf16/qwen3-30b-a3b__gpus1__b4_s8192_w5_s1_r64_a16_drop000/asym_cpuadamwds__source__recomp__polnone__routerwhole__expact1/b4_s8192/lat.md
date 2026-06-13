# LF LoRA-SFT Latency

| Metric | ms |
|---|---:|
| trainer e2e measured step incl optimizer | 107290.856 |
| trainer e2e total step incl warmup | 132472.753 |
| optimizer/update side = e2e measured - fwd/bwd | 1399.262 |
| lf.training_step.total | 105935.268 |
| step.forward + step.backward | 105891.594 |
| step.forward | 1961.852 |
| step.backward | 103929.742 |
| lf.grad_clip | 5.866 |
| lf.optimizer.step substage | 1008.193 |
| lf.scheduler.step | 0.260 |

Timing source: `heartbeat_dataloader_interval`
