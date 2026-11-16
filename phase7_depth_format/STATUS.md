# Camera-frame and scale study: training launched

16 November 2026. User-approved cloud cap: US$8.

The 972 transformed inputs passed RGB, labels, proper-rotation, ray and
relative-depth checks. A 40-epoch training-only T4 timing test passed.
Estimated total including a50% training-time margin and overhead reserve is
about$7.23; actual provider charges are not yet known.

Nine10000epoch jobs have been submitted. This is not a completed result.
No accuracy or improvement is claimed while training is running.

Cloud application:
https://modal.com/apps/shirsamitra22/main/ap-dVtAGklMO3lKAygdynLAK9

Read existing progress without starting a GPU:

```powershell
$env:PYTHONUTF8='1'
& .\.venv-modal\Scripts\python.exe phase7_depth_format/status.py
```

The first timing attempt completed its calculation but returned a TorchVersion
object that the lightweight local Modal client could not deserialize. It was
corrected to return a plain string and the short timing test repeated. This
engineering overhead is covered by the reserve; no held-out accuracy informed
the change. No full training job is automatically retried.

Original phase-six results, raw inputs, cancelled phase-five checkpoints and
the user's modified PDFs remain intact.

