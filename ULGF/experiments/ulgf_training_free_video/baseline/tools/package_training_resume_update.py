"""Package only the requested training changes, without model/data files."""
from pathlib import Path
import json
import zipfile
from .apply_ruod_visual_families import digest_file

FILES=['train_UWLGM.py','configs/data/ruod_256x256.py','tools/dist_train.sh',
       'ulgf_baseline/reviewed_data.py','ulgf_baseline/training_runtime.py',
       'tests/test_training_runtime.py']
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(root/'colab/ULGF_validation_resume_update.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(root/name,name)
        archive.writestr('update_manifest.json',json.dumps({n:digest_file(root/n) for n in FILES},indent=2))
    print('Training update ZIP built')
