"""Recover audited v11 images and an isolated pinned runtime; never train.

This implements prerequisite recovery, not unresolved visual-review decisions.
Every independent stage is attempted and persisted; model gates remain explicit.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from urllib.request import urlopen

INSTALLER = 'https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh'
INSTALLER_SHA = 'fc96109ea96493e31f70abbc5cae58e80634480c0686ab46924549ac41176812'
MMCV_INDEX = 'https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html'
TORCH_INDEX = 'https://download.pytorch.org/whl/cu113'
TORCH = ['torch==1.12.1+cu113', 'torchvision==0.13.1+cu113', 'torchaudio==0.12.1+cu113']
# One OpenCV distribution, one COCO distribution. Do not install legacy
# mmpycocotools beside pycocotools or headless beside desktop OpenCV.
DETECTION = ['numpy==1.21.6', 'Pillow==9.5.0', 'mmcv-full==1.7.0',
             'mmdet==2.25.3', 'opencv-python==4.10.0.84', 'pycocotools==2.0.7',
             'yapf==0.40.1', 'matplotlib==3.5.3', 'scipy==1.7.3']
TRAINING = ['accelerate==0.20.3', 'bbox-visualizer==0.1.0', 'diffusers==0.4.1',
            'huggingface-hub==0.16.4', 'transformers==4.24.0', 'ftfy==6.1.1',
            'tensorboard==2.11.2', 'protobuf==3.20.3']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def verify_base(base, spec):
    from apply_ruod_visual_families import contract
    for split, expected in spec['base_contracts'].items():
        data = json.loads((base/'annotations'/('instances_'+split+'.json')).read_text())
        if contract(data) != expected: raise ValueError('Changed v11 split: '+split)
    if sha(base/'annotations/instances_test.json') != spec['official_test_sha256']:
        raise ValueError('Official test bytes changed')
    if Path(json.loads((base/'paths.json').read_text())['image_prefix']) != Path('/content/ulgf-ruod-derived-v1/images'):
        raise ValueError('Unexpected derived root')


def resolve_base(repo, spec):
    runtime = Path('/content/ulgf-ruod-split-v11-reviewed')
    backup = repo/'outputs/ruod-visual-families-v11-expanded30/split'
    # Do not replace an existing, inconsistent runtime with a different copy.
    if runtime.exists():
        verify_base(runtime, spec)
        return runtime
    verify_base(backup, spec)
    staging = Path(tempfile.mkdtemp(prefix='ulgf-v11-restore-', dir='/content'))/'split'
    shutil.copytree(backup, staging)
    verify_base(staging, spec)
    staging.rename(runtime)
    return runtime


def main(repo, bundle):
    repo, bundle = Path(repo), Path(bundle)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    local = Path(tempfile.mkdtemp(prefix='ulgf-recovery-', dir='/content'))
    persistent = repo/'outputs/recovery-and-preflight'/stamp
    persistent.mkdir(parents=True, exist_ok=False)
    report = dict(scope='v11 prerequisite recovery and diagnostic qualification',
                  recorded_utc=stamp, recovery={}, training_ready=False,
                  training_started=False, pilot_started=False, membership_changed=False)
    prefix = Path('/content/ulgf-training-py37')
    python = prefix/'bin/python'
    spec = json.loads((bundle/'recovery_manifest.json').read_text())
    env = dict(os.environ, PYTHONNOUSERSITE='1')
    for key in ('PYTHONPATH', 'PYTHONHOME'): env.pop(key, None)

    def save():
        text = json.dumps(report, indent=2)
        (local/'report.json').write_text(text, encoding='utf-8')
        temporary = persistent/'report.pending.json'
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(persistent/'report.json')

    def run(name, args, timeout=3600):
        print(name+' ...', flush=True)
        log = local/(name+'.log')
        try:
            with log.open('w', encoding='utf-8') as stream:
                process = subprocess.Popen(args, cwd=str(repo), env=env,
                    stdout=stream, stderr=subprocess.STDOUT, text=True)
                started = time.monotonic()
                try:
                    while True:
                        remaining = timeout-(time.monotonic()-started)
                        if remaining <= 0: raise subprocess.TimeoutExpired(args, timeout)
                        try:
                            code = process.wait(timeout=min(30,remaining))
                            break
                        except subprocess.TimeoutExpired:
                            print(name+' still running; log: '+str(log),flush=True)
                except BaseException:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    raise
            if code:
                raise RuntimeError('{} exit {}: {}'.format(name, code, log.read_text(errors='replace')[-2500:]))
        finally:
            if log.exists(): shutil.copy2(log, persistent/log.name)
        return log.read_text(errors='replace')

    def stage(name, action):
        try: report['recovery'][name] = dict(status='PASS', details=action())
        except Exception as error: report['recovery'][name] = dict(status='FAIL', error=str(error))
        save()
        print(name+': '+report['recovery'][name]['status'], flush=True)

    def images():
        base = resolve_base(repo, spec)
        # Recovery uses the notebook's existing Pillow; installation isolation
        # below never modifies the notebook interpreter.
        log = run('image_recovery', [sys.executable, str(bundle/'restore_ruod_images.py'),
            '--source', '/content/drive/MyDrive/ULGF-assets/datasets/RUOD',
            '--base', str(base), '--bundle', str(bundle),
            '--output', '/content/ulgf-ruod-derived-v1/images'], timeout=21600)
        return dict(required=12557, base=str(base), result=log[-1200:])

    def runtime():
        if prefix.exists() and not python.is_file():
            raise RuntimeError('Partial isolated environment exists; preserved for inspection: '+str(prefix))
        if not python.is_file():
            installer = local/'miniconda37.sh'
            with urlopen(INSTALLER, timeout=120) as src, installer.open('wb') as out:
                shutil.copyfileobj(src, out)
            if sha(installer) != INSTALLER_SHA: raise ValueError('Installer SHA-256 mismatch')
            run('install_python', ['bash', str(installer), '-b', '-p', str(prefix)])
        run('python_version', [str(python), '-c',
            'import sys; assert sys.version_info[:3]==(3,7,16), sys.version; print(sys.version)'])
        # A dedicated prefix avoids uninstalling shared cv2/COCO distributions.
        run('conflict_check', [str(python), '-c',
            "import pkg_resources as p; names={d.key for d in p.working_set}; "
            "assert not names.intersection({'mmcv','opencv-python-headless','opencv-contrib-python','opencv-contrib-python-headless','mmpycocotools'}), names"])
        run('pip_bootstrap', [str(python), '-m', 'pip', 'install', 'pip==22.3.1', 'setuptools==65.6.3', 'wheel==0.38.4'])
        run('torch_install', [str(python), '-m', 'pip', 'install', '--only-binary=:all:',
            '--extra-index-url', TORCH_INDEX]+TORCH)
        run('opencv_system_libraries', [str(python), '-c',
            "import ctypes; ctypes.CDLL('libGL.so.1'); ctypes.CDLL('libgthread-2.0.so.0'); print('PASS')"])
        # Resolve the compiled wheel explicitly; no surprise source build.
        run('detection_install', [str(python), '-m', 'pip', 'install', '--only-binary=:all:',
            '--find-links', MMCV_INDEX]+DETECTION)
        import runpy
        probe = runpy.run_path(str(bundle/'colab_training_preflight.py'))['PROBE']
        # Test compiled ops before installing the remaining training packages.
        ops_probe = probe.split('    # Import the patched trainer')[0]+'\nprint(json.dumps(report))\n'
        ops = json.loads(run('detection_ops', [str(python), '-c', ops_probe]).strip().splitlines()[-1])
        if ops['compiled_ops'] != 'CPU_AND_CUDA_PASS': raise ValueError('Compiled operators not validated')
        constraints = local/'constraints.txt'
        constraints.write_text('\n'.join(TORCH+DETECTION+TRAINING)+'\n')
        run('training_install', [str(python), '-m', 'pip', 'install', '-c', str(constraints)]+TRAINING)
        run('pip_check', [str(python), '-m', 'pip', 'check'])
        run('pip_freeze', [str(python), '-m', 'pip', 'freeze'])
        result = json.loads(run('trainer_and_ops', [str(python), '-c', probe]).strip().splitlines()[-1])
        return dict(python=str(python), probe=result)

    try:
        stage('audited_image_recovery', images)
        stage('pinned_runtime_recovery', runtime)
        from consolidated_preflight import main as diagnose
        diagnostics = diagnose(repo, python, emit=False)
        report['checks'] = diagnostics['checks']
        report['blocking_gates'] = diagnostics['blocking_gates']
        report['checkpoint_candidates'] = diagnostics.get('checkpoint_candidates', {})
        report['training_ready'] = diagnostics['training_ready'] and all(
            x['status']=='PASS' for x in report['recovery'].values())
    except BaseException as error:
        report['interruption_or_error'] = str(error) or type(error).__name__
        raise
    finally:
        report['status'] = 'READY' if report['training_ready'] else 'NOT_READY'
        save()
        if sha(local/'report.json') != sha(persistent/'report.json'):
            raise ValueError('Persistent report readback mismatch')
        print('FINAL REPORT:', persistent/'report.json')
        print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
