"""One upload/cell for real prerequisite recovery, bound to audited v11."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
from .consolidated_preflight import CONTRACTS, TEST_SHA


def main():
    repo = Path(__file__).resolve().parents[1]
    source_spec = json.loads((repo/'outputs/ruod-v11-refresh-inputs/spec.json').read_text())
    manifest = dict(base_contracts=CONTRACTS, official_test_sha256=TEST_SHA,
                    source_annotation_hashes=source_spec['source_annotation_hashes'], records={})
    for origin in ('train', 'test'):
        splits = ('test',) if origin=='test' else ('train', 'validation')
        ids = set()
        for split in splits:
            data = json.loads((repo/'outputs/ruod-v11-refresh-base/annotations'/('instances_'+split+'.json')).read_text())
            payload = {k: sorted(data[k], key=lambda x:x['id']) for k in ('images','annotations','categories')}
            if hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=CONTRACTS[split]:
                raise ValueError('Local v11 evidence changed: '+split)
            ids.update(i['id'] for i in data['images'])
        audit = json.loads((repo/'outputs/ruod-local-audit/audit'/(origin+'.json')).read_text())
        manifest['records'][origin] = {str(r['id']): {k:r[k] for k in
            ('file_name','file_sha256','stored_pixels_sha256')} for r in audit if r['id'] in ids}
        if len(manifest['records'][origin]) != len(ids): raise ValueError('Recovery evidence incomplete')
    if sum(map(len,manifest['records'].values())) != 12557: raise ValueError('Wrong image count')
    names = ('recover_training_prerequisites.py','restore_ruod_images.py','audit_ruod_dataset.py',
             'apply_ruod_visual_families.py','consolidated_preflight.py','colab_training_preflight.py')
    archive = repo/'colab/ULGF_recovery_and_checks.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:
            source = (repo/'tools'/name).read_bytes()
            ast.parse(source)
            z.writestr(name,source)
        z.writestr('recovery_manifest.json',json.dumps(manifest))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    cell = '''# ONE cell: audited image recovery, isolated Python/dependencies, then checks.
# Select a GPU runtime before running. This does NOT launch model training.
from google.colab import drive, files
from pathlib import Path
import hashlib, io, json, subprocess, sys, tempfile, zipfile
drive.mount('/content/drive')
print('Upload ULGF_recovery_and_checks.zip, not the Python cell.')
uploaded=files.upload()
if len(uploaded)!=1: raise RuntimeError('Upload exactly one package')
blob=next(iter(uploaded.values()))
if hashlib.sha256(blob).hexdigest()!=EXPECTED_SHA:
    raise RuntimeError('Wrong or damaged recovery package; nothing extracted')
workspace=Path(tempfile.mkdtemp(prefix='ulgf-recovery-tool-',dir='/content'))
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    if set(archive.namelist())!=EXPECTED_NAMES or len(archive.namelist())!=len(EXPECTED_NAMES):
        raise RuntimeError('Unexpected package contents')
    archive.extractall(workspace)
repo=Path('/content/drive/MyDrive/ULGF-main')
if not (repo/'train_UWLGM.py').is_file(): raise FileNotFoundError(repo)
result=subprocess.run([sys.executable,str(workspace/'recover_training_prerequisites.py'),str(repo),str(workspace)])
if result.returncode: raise RuntimeError('Recovery interrupted; inspect its saved report. Existing verified images are reusable.')
'''.replace('EXPECTED_SHA',repr(digest)).replace('EXPECTED_NAMES',repr(set(names)|{'recovery_manifest.json'}))
    ast.parse(cell)
    (repo/'colab/ULGF_recovery_and_checks_cell.py').write_text(cell,encoding='utf-8')
    print('Packaged 12,557 hash-bound recovery records. SHA-256:',digest)


if __name__=='__main__': main()
