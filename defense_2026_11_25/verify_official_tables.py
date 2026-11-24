"""Independent canonical AcroForm and annotation checks with pypdf."""
from pathlib import Path
import hashlib
import json
from pypdf import PdfReader

HOME = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((HOME / 'official_tables_manifest.json').read_text())
    target = Path(manifest['output'])
    assert sha(target) == manifest['output_sha256']
    assert sha(Path(manifest['source'])) == manifest['source_sha256']
    reader = PdfReader(target)
    fields = reader.get_fields() or {}
    expected = {r['name']: r for r in manifest['fields']}
    assert len(reader.pages) == 10 and len(fields) == len(expected) == 63
    assert set(fields) == set(expected)
    widgets = {}
    for page_number, page in enumerate(reader.pages, 1):
        for ref in page.get('/Annots', []):
            annotation = ref.get_object()
            if annotation.get('/Subtype') != '/Widget':
                continue
            field = annotation
            while '/T' not in field and '/Parent' in field:
                field = field['/Parent'].get_object()
            name = str(field['/T'])
            assert name not in widgets
            assert name in expected and expected[name]['page'] == page_number
            assert int(annotation.get('/F', 0)) & 4
            assert '/AP' in annotation and '/N' in annotation['/AP']
            actual = str(fields[name].get('/V', '')).removeprefix('/')
            widget_value = str(field.get('/V', '')).removeprefix('/')
            assert actual == widget_value == expected[name]['value']
            if expected[name]['owner'] == 'supervisor':
                assert actual in ('', 'Off')
            assert str(fields[name].get('/FT')) != '/Sig'
            widgets[name] = page_number
    assert set(widgets) == set(fields)
    assert fields['t1_title']['/V'] == fields['t3_title']['/V'] == manifest['title_unchanged']
    assert fields['t1_student_id']['/V'] == fields['t3_student_id']['/V'] == '2023521460130'
    assert fields['t1_supervisor_name']['/V'] == fields['t3_supervisor_name']['/V'] == 'SHAOBING GAO'
    teacher_count = sum(r['owner'] == 'supervisor' for r in expected.values())
    result = {'status': 'PASS_PREFILLED_UNSIGNED', 'pages': len(reader.pages), 'fields_checked': len(fields),
              'canonical_and_widget_values_match': True, 'all_widgets_printable': True,
              'teacher_fields_left_blank': teacher_count, 'signatures_added': 0,
              'topic_changed': False, 'official_source_unchanged': True,
              'output_sha256': sha(target), 'verifier_sha256': sha(Path(__file__))}
    (HOME / 'official_tables_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
