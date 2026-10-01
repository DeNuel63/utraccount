"""Capture the existing source baseline without modifying its files."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    target = ROOT/'baseline'
    if target.exists():
        raise RuntimeError('Baseline already exists; refusing to replace it')
    paths = []
    for folder in ('ulgf_baseline','ulgf_video','pipeline_custom','utils','configs','scripts','tests','tools'):
        paths.extend(p for p in (REPO/folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix in ('.py','.sh','.json','.yaml','.yml'))
    paths.extend(REPO/p for p in ('train_UWLGM.py','requirements.txt','requirements-inference.txt',
                                 'ULGF_VIDEO_ENGINEERING_ROADMAP.md','docs/video_output_schema.md'))
    paths.extend(p for p in (REPO/'assets/smoke').rglob('*') if p.is_file())
    records={}
    for source in sorted(set(paths)):
        relative=source.relative_to(REPO).as_posix()
        dest=target/relative; dest.parent.mkdir(parents=True,exist_ok=True)
        before=digest(source); shutil.copy2(source,dest)
        if digest(dest)!=before or digest(source)!=before: raise RuntimeError('Snapshot mismatch')
        records[relative]=before
    (ROOT/'baseline_snapshot.json').write_text(json.dumps(dict(
        source='../../', files=records, weights_copied=False,
        purpose='Independent source snapshot; never import live training code'),indent=2))
    print('Snapshot files:',len(records))

if __name__=='__main__': main()
