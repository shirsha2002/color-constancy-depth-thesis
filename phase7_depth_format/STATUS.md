# Camera-frame and scale study: completed and audited

16 November 2026. Nine new released-head PCCC models completed10000epochs on the official NYU/DIODE folds. Read RESULTS.md and results/nyu_frame_v1/audit.json for all measured results and limits. No reliability gate has been implemented.

Training time: 5.758 GPU-hours. Training compute planning allowance (GPU+maximum CPU/RAM): $4.67; smoke/startup/build overhead is additional. This is not an invoice. User-approved cap was$8. No extra training or automatic retry is pending. The cancelled laptop study remains stopped.

An initial short timing run returned a TorchVersion object incompatible with the lightweight client; it was changed to a plain string and the timing test repeated. This overhead is covered by the reserve.
