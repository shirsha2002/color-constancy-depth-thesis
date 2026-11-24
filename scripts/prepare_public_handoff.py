"""Prepare public presentation assets without copying private meeting material."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / 'defense_2026_11_25'
OUT = ROOT / 'output/defense_2026_11_25'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    figures = OUT / 'figures'
    previews = OUT / 'slide_previews'
    figures.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    names = ['input_clouds', 'training_curves', 'noise_results', 'illuminant_outputs']
    for name in names:
        source = HOME / f'.build/experiment_report_figures/{name}.png'
        if source.exists():
            shutil.copy2(source, figures / source.name)
        assert (figures / f'{name}.png').is_file()
    for number in range(1, 8):
        source = HOME / f'.build/powerpoint-final/Slide{number}.PNG'
        target = previews / f'slide-{number:02}.png'
        if source.exists():
            shutil.copy2(source, target)
        assert target.is_file()
    files = [ROOT / 'FRIEND_START_HERE.md'] + [OUT / n for n in [
        'Shirsha_Mitra_Proposal_Defense_2026.pptx',
        'Shirsha_Mitra_Proposal_Report_With_Experiments.docx',
        'Shirsha_Mitra_Proposal_Report_With_Experiments.pdf',
        'Three_Minute_Script_English_and_Bengali.txt', 'Defense_QA_English_and_Bengali.txt', 'QQ_Draft_English.txt']]
    files += list(figures.glob('*.png')) + list(previews.glob('*.png'))
    files += [ROOT / 'output/proposal_submission/2023521460130_Shirsha Mitra_Proposal Report Table1_3.pdf',
              ROOT / 'output/proposal_submission/README.txt']
    files += [ROOT / 'phase9_reliability_fusion/pilot_v1' / n for n in ['RESULTS.md', 'summary.csv', 'summary.json', 'predictions.json', 'paired.json', 'audit.json', 'independent_verification.json']]
    files += [ROOT / 'phase9_reliability_fusion' / n for n in ['README.md', 'AUDIT_RECOVERY.md', 'PILOT_PROTOCOL.md']]
    files += [ROOT / 'output/pdf' / n for n in ['Shirsha_Mitra_PCCC_GPU_Update_2026_11_10.pdf', 'Shirsha_Mitra_PCCC_Depth_Format_Update_2026_11_16.pdf']]
    values = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    assert len(values) == len(files)
    manifest = {'prepared_date': '2026-11-24', 'purpose': 'Public thesis proposal presentation handoff',
                'files_sha256': values, 'main_slides': 5, 'backup_slides': 2,
                'research_status': 'Completed reduced-budget pilot; no established gating benefit',
                'official_forms_status': 'Prefilled unsigned; supervisor completion required',
                'private_meeting_information_included': False}
    manifest_path = OUT / 'Public_File_Hashes.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    archive = OUT / 'Presentation_Materials_Public.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.write(manifest_path, manifest_path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, digest in values.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
        assert not any('/private/' in n or 'Defense_Pack_Private' in n for n in z.namelist())
    print(json.dumps({'status': 'PASS', 'files': len(files), 'zip_mib': round(archive.stat().st_size/1048576, 2),
                      'zip_sha256': sha(archive), 'public_zip': str(archive)}, indent=2))


if __name__ == '__main__':
    main()
