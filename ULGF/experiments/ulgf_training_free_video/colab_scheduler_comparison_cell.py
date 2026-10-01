"""Paste into one isolated Colab cell. Upload the comparison ZIP when prompted."""
from google.colab import drive, files
drive.mount('/content/drive')
import io, zipfile, subprocess, tempfile, os
from pathlib import Path
from datetime import datetime, timezone
from IPython.display import display, Image

root=Path('/content/drive/MyDrive/ulgf_training_free_video')
python=Path('/content/ulgf-video-py37/bin/pyt@
hon')
checkpoint=Path('/content/drive/MyDrive/ULGF-assets/checkpoints/ruod_256_bz16/checkpoint')
for path in (python,root/'run_five_frames.py',checkpoint):
    if not path.exists():raise FileNotFoundError('Restore isolated environment/input: '+str(path))
print('Upload ULGF_scheduler_comparison.zip. GPU is needed for the four-image comparison.')
uploaded=files.upload()
if len(uploaded)!=1:raise ValueError('Upload exactly one ZIP')
with zipfile.ZipFile(io.BytesIO(next(iter(uploaded.values())))) as bundle:
    expected={'scheduler_control.py','run_scheduler_comparison.py','test_scheduler_control.py'}
    if set(bundle.namelist())!=expected or len(bundle.namelist())!=3:
        raise ValueError('Unexpected ZIP contents')
    work=Path(tempfile.mkdtemp(prefix='ulgf-scheduler-ab-'))
    bundle.extractall(work)
env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1')
env['ULGF_EXPERIMENT_ROOT']=str(root)
subprocess.run([str(python),'-m','unittest','test_scheduler_control','-v'],cwd=str(work),env=env,check=True)
name='scheduler_ab_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
process=subprocess.Popen([str(python),'-u',str(work/'run_scheduler_comparison.py'),
    '--checkpoint',str(checkpoint),'--run-name',name],cwd=str(work),env=env,
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
try:
    for line in process.stdout:print(line,end='',flush=True)
    code=process.wait()
finally:
    if process.poll() is None:
        process.terminate();process.wait()
if code:raise RuntimeError('Comparison failed; preserve outputs and send the error.')
display(Image(filename=str(root/'runs'/name/'comparison.png')))
print('Send comparison.png and report.json from:',root/'runs'/name)
