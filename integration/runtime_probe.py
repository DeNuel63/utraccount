"""Inspect one stage's installed runtime without loading or changing weights."""
import argparse
import importlib
import json
import platform
import sys

MODULES = {
    'ulgf': ['torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'cv2', 'ftfy', 'bbox_visualizer'],
    'countgd': ['torch', 'torchvision', 'transformers', 'detectron2', 'gradio', 'MultiScaleDeformableAttention'],
    'covtrack': ['torch', 'torchvision', 'mmcv', 'mmcv.ops', 'mmdet', 'clip', 'lvis', 'motmetrics', 'seaborn', 'h5py'],
}


def probe(stage):
    report = dict(stage=stage, executable=sys.executable, python=platform.python_version(),
        modules={}, cuda_available=False, status='IMPORT_PROBE_FAILED', model_loaded=False,
        model_compatibility_verified=False)
    for name in MODULES[stage]:
        try:
            module = importlib.import_module(name)
            report['modules'][name] = dict(imported=True, version=getattr(module, '__version__', None))
            if name == 'torch':
                report['cuda_available'] = module.cuda.is_available()
                report['torch_cuda_build'] = module.version.cuda
                if report['cuda_available']:
                    report['gpu'] = module.cuda.get_device_name(0)
        except Exception as error:
            report['modules'][name] = dict(imported=False, error=str(error))
    if all(item['imported'] for item in report['modules'].values()) and report['cuda_available']:
        report['status'] = 'IMPORT_PROBE_PASS'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=sorted(MODULES))
    args = parser.parse_args()
    result = probe(args.stage)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'IMPORT_PROBE_PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
