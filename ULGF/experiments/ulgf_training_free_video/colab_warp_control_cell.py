# Run this as one cell in the separate experiment notebook. CPU only.
# Upload run_warp_control.py beside run_five_frames.py in the experiment first.
from google.colab import drive
from pathlib import Path
from datetime import datetime, timezone
from IPython.display import display, Image as DisplayImage
import json
import subprocess
import sys

drive.mount('/content/drive')
root=Path('/content/drive/MyDrive/ulgf_training_free_video')
reference=root/'runs/five_frame_20260928T011622456925Z'
script=root/'run_warp_control.py'
for path in (script,reference/'manifest.json',reference/'run.json'):
    if not path.is_file():raise FileNotFoundError(path)
name='warp_control_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
output=root/'runs'/name
print('CPU-only warping. No model, GPU or training is used.',flush=True)
result=subprocess.run([sys.executable,str(script),'--reference-run',str(reference),
    '--run-name',name],cwd=str(root),text=True,stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,timeout=300)
print(result.stdout)
if result.returncode:raise RuntimeError('Control stopped. Preserve and send the error; do not overwrite the reference.')
report=json.loads((output/'qc.json').read_text())
print('RESULT:',report['status'])
print('Animal appearance approved:',report['animal_resemblance_approved'])
print('Foreground pixels preserved:',all(f['source_mask_pixels_preserved'] for f in report['frames']))
print('Saved:',output)
display(DisplayImage(filename=str(output/'comparison.png'),width=1280))
print('Red overlay: selected region. It is not an automatically verified animal mask.')
display(DisplayImage(filename=str(output/'diagnostics/source_mask_overlay.png')))
print('Background repair: inspect for visible smears, seams or remaining animal fragments.')
display(DisplayImage(filename=str(output/'diagnostics/background_fill.png')))
