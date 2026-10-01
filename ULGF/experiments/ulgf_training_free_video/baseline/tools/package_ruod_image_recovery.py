"""Build hash-bound image-only recovery package from saved audit evidence."""
import json
from pathlib import Path
import zipfile

repo = Path(__file__).resolve().parents[1]
manifest = json.loads((repo / 'outputs/ruod-visual-family-proposal-batch03/reviewed_visual_families.json').read_text())
spec = {key: manifest[key] for key in ('base_contracts', 'source_annotation_hashes')}
spec['official_test_sha256'] = '554b3a631fccf53f651aee8b8e0fcccf2c4a3763325d86ab9807449a7f4cff8f'
spec['records'] = {}
for split in ('train', 'test'):
    records = json.loads((repo / ('outputs/ruod-local-audit/audit/' + split + '.json')).read_text())
    spec['records'][split] = {str(r['id']): {k: r[k] for k in
        ('file_name', 'file_sha256', 'stored_pixels_sha256')} for r in records}
with zipfile.ZipFile(repo / 'colab/RUOD_image_recovery.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ('restore_ruod_images.py', 'audit_ruod_dataset.py', 'apply_ruod_visual_families.py'):
        archive.write(repo / 'tools' / name, name)
    archive.writestr('recovery_manifest.json', json.dumps(spec))
print('Built colab/RUOD_image_recovery.zip')
