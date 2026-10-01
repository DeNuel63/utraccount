"""One isolated GPU cell. Final bounded source-crop comparison."""
from google.colab import drive, files
drive.mount('/content/drive')
from pathlib import Path
from datetime import datetime, timezone
import io, zipfile, tempfile, subprocess, os
from IPython.display import display, Image
root=Path('/content/drive/MyDrive/ulgf_training_free_video')
python=Path('/content/ulgf-video-py37/bin/python')
checkpoint=Path('/content/drive/MyDrive/ULGF-assets/checkpoints/ruod_256_bz16/checkpoint')
for path in (python,root/'run_five_frames.py',checkpoint):
    if not path.exists():raise FileNotFoundError('Restore isolated environment/input: '+str(path))
print('Upload ULGF_fish_crop_comparison.zip, not this cell file.')
uploaded=files.upload()
if len(uploaded)!=1:raise ValueError('Upload exactly one ZIP')
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as bundle:
    if len(bundle.namelist())!=2 or set(bundle.namelist())!={'image_conditioning.py','run_image_conditioning.py'}:
        raise ValueError('Unexpected archive contents')
    work=Path(tempfile.mkdtemp(prefix='ulgf-fish-crop-'));bundle.extractall(work)
name='fish_crop_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1')
process=subprocess.Popen([str(python),'-u',str(work/'run_image_conditioning.py'),
    '--checkpoint',str(checkpoint),'--run-name',name,'--fish-crop'],cwd=str(work),env=env,
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
try:
    for line in process.stdout:print(line,end='',flush=True)
    code=process.wait()
finally:
    if process.poll() is None:process.terminate();process.wait()
if code:raise RuntimeError('Run failed; retain outputs and send the error. No automatic retry.')
display(Image(filename=str(root/'runs'/name/'comparison.png')))
archive=Path('/content')/(name+'_review.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
    for filename in ('report.json','comparison.png','source_prepared.png','VAE_reconstruction.png','strength_0.05.png','strength_0.1.png'):
        bundle.write(root/'runs'/name/filename,arcname=filename)
print('Downloading review bundle; upload it here. Source and training remain unchanged.')
files.download(str(archive))
