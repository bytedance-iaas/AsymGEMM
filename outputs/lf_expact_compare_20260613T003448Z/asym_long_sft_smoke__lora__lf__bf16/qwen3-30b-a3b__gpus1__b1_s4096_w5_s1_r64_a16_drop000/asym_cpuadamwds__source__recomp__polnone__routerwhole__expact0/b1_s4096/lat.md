# LF LoRA-SFT Latency

| Metric | ms |
|---|---:|
| trainer e2e measured step incl optimizer | 3128.213 |
| trainer e2e total step incl warmup | 7715.115 |
| optimizer/update side = e2e measured - fwd/bwd | 1341.578 |
| lf.training_step.total | 1827.363 |
| step.forward + step.backward | 1786.635 |
| step.forward | 591.154 |
| step.backward | 1195.481 |
| lf.grad_clip | 5.259 |
| lf.optimizer.step substage | 995.697 |
| lf.scheduler.step | 0.271 |

Timing source: `heartbeat_dataloader_interval`
