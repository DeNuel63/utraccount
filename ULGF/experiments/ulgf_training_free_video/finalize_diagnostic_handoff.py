"""Create a versioned, self-contained diagnostic bundle without changing prior runs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,zipfile
from verify_delivery import verify

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/sequence_delivery_20260930T214150947571Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    previous={str(p):sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    dest=ROOT/'runs'/('final_diagnostic_handoff_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    dest.mkdir(parents=True,exist_ok=False)
    for name in ('sequence_5frame','sequence_48frame'):shutil.copytree(SOURCE/name,dest/name)
    shutil.copy2(ROOT/'BRANCH_FINAL_REPORT.md',dest/'BRANCH_FINAL_REPORT.md')
    shutil.copy2(ROOT/'verify_delivery.py',dest/'verify_delivery.py')
    verify(dest)
    evidence=dict(status='DIAGNOSTIC_HANDOFF_COMPLETE',normal_training_completed=False,
        learned_video_generation_claim=False,dataset_accepted=False,verified_frames=53,
        five_frame_user_accepted=True,longer_trajectory_user_approval='PENDING',
        verifier_sha256=sha(dest/'verify_delivery.py'),report_sha256=sha(dest/'BRANCH_FINAL_REPORT.md'))
    (dest/'HANDOFF_STATUS.json').write_text(json.dumps(evidence,indent=2))
    sums={str(p.relative_to(dest)).replace('\\','/'):sha(p) for p in dest.rglob('*') if p.is_file()}
    (dest/'BUNDLE_SHA256SUMS.json').write_text(json.dumps(sums,indent=2))
    archive=dest.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in dest.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(dest)))
    with zipfile.ZipFile(archive) as z:
        for name,h in sums.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    assert all(sha(Path(p))==h for p,h in previous.items())
    print('FINAL BUNDLE:',archive.name)
    print('ZIP SHA256:',sha(archive))
    print('Prior delivery files unchanged; archive hashes verified.')

if __name__=='__main__':main()
