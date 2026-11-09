# Paper-budget reproduction cancelled

Cancelled at the user's explicit request on 9 November 2026 because the run takes too much time.

Both the CPU reference and Intel XPU suite workers were stopped. The remaining queued jobs will not run, and the completion publisher will not run. No10000epoch result is claimed.

Latest complete checkpoints are preserved locally:

- xpu_full_v1: epoch 6200 /10000.
- cpu_reference_v1: epoch 1550 /10000.

Earlier completed pilot results and source remain intact. Resumption requires an explicit new instruction; nothing is scheduled to restart automatically.
