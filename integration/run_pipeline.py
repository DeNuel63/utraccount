"""Launch existing ULGF inference, CountGD++, then COVTrack in separate runtimes."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.common import read_clip, validate_class_map, write_json


def resolve_config(config, base):
    value = json.loads(json.dumps(config))
    def path(text):
        p = Path(text).expanduser()
        return str((base / p).resolve() if not p.is_absolute() else p.resolve())
    value['output'] = path(value['output'])
    if value.get('clip_manifest'):
        value['clip_manifest'] = path(value['clip_manifest'])
    for stage in ('countgd', 'covtrack', 'ulgf'):
        if stage not in value:
            continue
        for key in ('python', 'checkpoint', 'config', 'repository'):
            if key in value[stage]:
                value[stage][key] = path(value[stage][key])
    if value.get('ulgf'):
        name = value['ulgf']['run_name']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('ULGF run_name must be a single new directory name')
        value['clip_manifest'] = str(ROOT / 'ULGF' / 'experiments' / 'ulgf_training_free_video' / 'runs' / name / 'manifest.json')
    return value


def preflight(config):
    classes = validate_class_map(config['classes'])
    if config['prompt'].strip() not in classes:
        raise ValueError('Prompt must match a class name exactly')
    threshold = config['confidence_threshold']
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError('Confidence threshold must be between zero and one')
    output = Path(config['output'])
    if output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(output))
    required = []
    for stage in ('countgd', 'covtrack'):
        required.extend([config[stage]['python'], config[stage]['checkpoint']])
    required.extend([config['countgd']['repository'] + '/app.py', config['covtrack']['config']])
    if config.get('ulgf'):
        required.extend([config['ulgf']['python'], config['ulgf']['checkpoint'] + '/model_index.json'])
        if Path(config['clip_manifest']).parent.exists():
            raise FileExistsError('Choose a new ULGF run_name')
    else:
        required.append(config['clip_manifest'])
    missing = [p for p in required if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError('Missing runtime inputs:\n' + '\n'.join(missing))
    protected = [Path(config['clip_manifest']).parent, Path(config['countgd']['checkpoint']), Path(config['covtrack']['checkpoint'])]
    if config.get('ulgf'):
        protected.append(Path(config['ulgf']['checkpoint']))
    for source in protected:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('Pipeline output overlaps an input')
    if not config.get('ulgf'):
        read_clip(config['clip_manifest'], classes)


def execute(config, dry_run=False, launch=subprocess.run):
    preflight(config)
    if dry_run:
        return dict(status='PREFLIGHT_PASS', model_execution_verified=False)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    resolved = output / 'resolved_config.json'
    write_json(resolved, config)
    report = dict(status='RUNNING', stages=[], model_execution_verified=False)
    write_json(output / 'status.json', report)
    try:
        if config.get('ulgf'):
            stage = config['ulgf']
            command = [stage['python'], str(ROOT / 'ULGF/experiments/ulgf_training_free_video/run_five_frames.py'),
                '--checkpoint', stage['checkpoint'], '--run-name', stage['run_name'],
                '--seed', str(stage.get('seed', 0)), '--steps', str(stage.get('steps', 100)),
                '--guidance', str(stage.get('guidance', 5.0))]
            launch(command, cwd=str(ROOT / 'ULGF'), check=True)
            report['stages'].append('ulgf')
        read_clip(config['clip_manifest'], config['classes'])
        for name, script, cwd in [('countgd', 'count_stage.py', ROOT), ('covtrack', 'track_stage.py', ROOT / 'CovTrack_Model/COVTrack')]:
            launch([config[name]['python'], str(ROOT / 'integration' / script), '--config', str(resolved), '--output', str(output)], cwd=str(cwd), check=True)
            report['stages'].append(name)
            write_json(output / 'status.json', report)
        if not (output / 'tracking_sequence.json').is_file():
            raise RuntimeError('Tracker did not produce its sequence output')
        report.update(status='COMPLETE', model_execution_verified=True)
    except Exception as error:
        report.update(status='FAILED', error=str(error))
        raise
    finally:
        write_json(output / 'status.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = resolve_config(json.loads(args.config.read_text(encoding='utf-8')), args.config.resolve().parent)
    print(json.dumps(execute(config, args.dry_run), indent=2))


if __name__ == '__main__':
    main()
