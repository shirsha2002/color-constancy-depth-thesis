"""Download public study ZIP parts and verify their SHA-256 hashes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'output/research_downloads/CHECKPOINT_DOWNLOADS.json'
RELEASES = 'https://github.com/shirsha2002/color-constancy-depth-thesis/releases/download'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset: dict, tag: str, destination: Path) -> None:
    name = asset['asset']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+\.zip', name):
        raise ValueError(f'Unexpected archive name: {name}')
    target = destination / name
    if target.exists() and target.stat().st_size == asset['bytes'] and sha256(target) == asset['sha256']:
        print(f'Already verified: {name}', flush=True)
        return
    partial = target.with_suffix('.zip.partial')
    url = f'{RELEASES}/{quote(tag, safe="")}/{quote(name, safe="")}'
    for attempt in range(3):
        try:
            request = Request(url, headers={'User-Agent': 'Thesis-Checkpoint-Downloader'})
            with urlopen(request, timeout=60) as response, partial.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if partial.stat().st_size != asset['bytes'] or sha256(partial) != asset['sha256']:
                raise ValueError(f'Checksum or size mismatch: {name}')
            partial.replace(target)
            print(f'Verified: {name}', flush=True)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    studies = [study['study'] for study in manifest['studies']]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List studies without downloading')
    parser.add_argument('--study', choices=studies, help='Download one study instead of all studies')
    parser.add_argument('--output', type=Path, default=ROOT / 'downloads/checkpoints')
    args = parser.parse_args()
    if args.list:
        for study in manifest['studies']:
            print(f"{study['study']}: {study['models']} models, {len(study['archives'])} ZIP parts")
        return
    assets = [a for a in manifest['assets'] if args.study is None or a['study'] == args.study]
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(assets)} parts ({sum(a['bytes'] for a in assets) / 1048576:.1f} MiB)", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda asset: download(asset, manifest['release_tag'], args.output), assets))
    print(f'All selected parts verified in {args.output.resolve()}')


if __name__ == '__main__':
    main()
