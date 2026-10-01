"""Readiness check only: does not install packages, load weights or train.

Run from the Colab notebook kernel; probes the isolated Python 3.7 subprocess.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import tempfile
import traceback


PROBE = r'''
import importlib.util
import json
import platform
import sys
import pkg_resources
import torch

assert sys.version_info[:3] == (3, 7, 16), sys.version
assert torch.__version__ == '1.12.1+cu113', torch.__version__
assert torch.version.cuda == '11.3', torch.version.cuda
assert torch.cuda.is_available(), 'GPU is unavailable'
# Prove CUDA arithmetic and backward are usable without allocating ULGF.
x = torch.ones(4, device='cuda', requires_grad=True)
(x.square().sum()).backward()
assert torch.equal(x.grad, torch.full_like(x, 2))
versions = {}
for package in ('mmcv', 'mmcv-full', 'mmdet', 'accelerate', 'diffusers', 'transformers'):
    try:
        versions[package] = pkg_resources.get_distribution(package).version
    except pkg_resources.DistributionNotFound:
        versions[package] = None
assert versions['mmcv'] is None, 'Standalone mmcv conflicts with mmcv-full; inspect installation first'
report = dict(python=platform.python_version(), torch=torch.__version__,
              cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
              cuda_backward='PASS', packages=versions, compiled_ops='NOT_INSTALLED')
if versions['mmcv-full'] is not None or versions['mmdet'] is not None:
    assert versions['mmcv-full'] == '1.7.0' and versions['mmdet'] == '2.25.3', versions
    import mmcv
    import mmdet
    from mmcv.ops import nms, roi_align
    from mmdet.datasets import build_dataset
    from pycocotools.coco import COCO
    for device in ('cpu', 'cuda'):
        boxes = torch.tensor([[0., 0., 4., 4.], [0., 0., 4., 4.], [8., 8., 10., 10.]], device=device)
        scores = torch.tensor([.9, .8, .7], device=device)
        _, keep = nms(boxes, scores, .5)
        assert keep.cpu().tolist() == [0, 2], keep
        features = torch.randn(1, 1, 8, 8, device=device, requires_grad=True)
        rois = torch.tensor([[0., 1., 1., 6., 6.]], device=device)
        roi_align(features, rois, (2, 2)).sum().backward()
        assert features.grad is not None and torch.isfinite(features.grad).all()
        assert features.grad.abs().sum() > 0
    torch.cuda.synchronize()
    report['compiled_ops'] = 'CPU_AND_CUDA_PASS'
    # Import the patched trainer without invoking main or loading model weights.
    import train_UWLGM
    report['trainer_import'] = 'PASS'
print(json.dumps(report))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--python', default='/content/miniconda37/bin/python')
    args = parser.parse_args()
    if not args.repo.is_dir():
        raise FileNotFoundError('Mount Drive and check repository: {}'.format(args.repo))
    output = args.repo / 'outputs' / 'training-preflight'
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')
    report_path = output / (stamp + '.json')
    report = dict(status='RUNNING', created_utc=stamp, installs_performed=False,
                  training_started=False, stages=[])

    def run(name, command, timeout=600):
        print(name + ' ...', flush=True)
        result = subprocess.run(command, cwd=str(args.repo), capture_output=True, text=True, timeout=timeout)
        report['stages'].append(dict(name=name, exit_code=result.returncode,
                                     stdout=result.stdout, stderr=result.stderr))
        if result.returncode:
            raise RuntimeError('{} failed:\n{}\n{}'.format(name, result.stdout, result.stderr))
        return result.stdout

    try:
        if not Path(args.python).is_file():
            raise FileNotFoundError('Python 3.7 runtime missing; restore the validated environment first')
        probe = json.loads(run('Runtime and installed-package checks', [args.python, '-c', PROBE]))
        report['runtime'] = probe
        with tempfile.TemporaryDirectory(prefix='ulgf-wheel-check-') as temporary:
            wheel_dir = Path(temporary)
            run('Resolve MMCV 1.7.0 binary wheel', [args.python, '-m', 'pip', 'download',
                '--no-cache-dir', '--no-deps', '--only-binary=:all:', '--no-index',
                '--find-links', 'https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html',
                '--dest', str(wheel_dir), 'mmcv-full==1.7.0'])
            run('Resolve MMDetection 2.25.3 wheel', [args.python, '-m', 'pip', 'download',
                '--no-cache-dir', '--no-deps', '--only-binary=:all:',
                '--dest', str(wheel_dir), 'mmdet==2.25.3'])
            report['resolved_wheels'] = sorted(p.name for p in wheel_dir.glob('*.whl'))
        if probe['compiled_ops'] == 'CPU_AND_CUDA_PASS':
            report['status'] = 'MMCV_MMDET_RUNTIME_PASS'
        else:
            report['status'] = 'WHEELS_AVAILABLE_INSTALL_AND_OPS_TEST_PENDING'
        report['remaining'] = [
            'Full dependency resolution and trainer import in Python 3.7',
            'Dataset transform integration tests on representative RUOD annotations',
            'Actual ULGF backward-pass VRAM and training-state resume test',
        ]
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: value for key, value in report.items() if key != 'stages'}, indent=2))
        print('Saved report:', report_path)
    if report['status'] == 'FAIL':
        raise RuntimeError(report['error'])


if __name__ == '__main__':
    main()
