"""Print useful setup information without inspecting credentials."""
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

def main():
    import torch
    versions = {}
    for package in ['torch', 'torchvision', 'opencv-python', 'numpy', 'scipy', 'matplotlib', 'h5py', 'requests', 'Pillow', 'reportlab', 'PyMuPDF']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    info = dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=versions,
                cuda_available=torch.cuda.is_available(), xpu_available=bool(hasattr(torch,'xpu') and torch.xpu.is_available()))
    print(json.dumps(info, indent=2))
    out=Path('results'); out.mkdir(exist_ok=True)
    (out/'environment.json').write_text(json.dumps(info,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
