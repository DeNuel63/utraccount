"""CPU preparation and offline GPU qualification; neither mode trains a model.

Dataset evidence and historical recovery helpers are supplied by a hash-bound
private package, not committed to the source repository.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from urllib.request import urlopen


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.pending')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def safe_target(root, name):
    relative = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe archive path: '+name)
    target = (Path(root)/name).resolve()
    target.relative_to(Path(root).resolve())
    return target


def publish_file(source, target):
    """Never overwrite a conflicting completed cache file."""
    source, target = Path(source), Path(target)
    expected = digest(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != expected: raise ValueError('Existing cache differs: '+str(target))
        return expected
    temp = target.with_name(target.name+'.pending')
    shutil.copyfile(source, temp)
    if digest(temp) != expected: raise ValueError('Cache copy readback failed')
    temp.replace(target)
    return expected


def make_shard(root, names, destination):
    files = {name: dict(sha256=digest(safe_target(root,name)),
                       size=safe_target(root,name).stat().st_size) for name in names}
    with tarfile.open(str(destination), 'w') as archive:
        for name in sorted(names):
            path=safe_target(root,name)
            info=archive.gettarinfo(str(path),arcname=name)
            info.mtime=0;info.uid=0;info.gid=0;info.uname='';info.gname=''
            info.mode=0o644
            with path.open('rb') as stream: archive.addfile(info,stream)
    return dict(sha256=digest(destination), files=files)


def restore_shard(archive_path, record, destination):
    """Extract only declared regular files and verify each; no extractall."""
    if digest(archive_path) != record['sha256']: raise ValueError('Shard hash mismatch')
    seen = set()
    with tarfile.open(str(archive_path),'r:') as archive:
        for member in archive:
            if not member.isfile() or member.name not in record['files'] or member.name in seen:
                raise ValueError('Unexpected/duplicate/nonregular member: '+member.name)
            info = record['files'][member.name]
            if member.size != info['size']: raise ValueError('Member size mismatch')
            path = safe_target(destination, member.name)
            seen.add(member.name)
            if path.exists():
                if digest(path) != info['sha256']: raise ValueError('Existing image differs: '+str(path))
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name+'.restore.pending')
            with archive.extractfile(member) as src, temp.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            if digest(temp) != info['sha256']: raise ValueError('Restored file hash mismatch')
            temp.replace(path)
    if seen != set(record['files']): raise ValueError('Missing archive members')


def check_ready(cache, spec_sha):
    path = Path(cache)/'READY.json'
    if not path.is_file(): raise ValueError('CPU preparation incomplete: READY.json missing. Keep GPU off.')
    ready = json.loads(path.read_text())
    if ready.get('status') != 'CPU_PREPARATION_PASS' or ready.get('evidence_sha256') != spec_sha:
        raise ValueError('Wrong or incomplete preparation record')
    for name, sha in ready['files'].items():
        if digest(safe_target(cache,name)) != sha: raise ValueError('Cache artifact changed: '+name)
    return ready


def main(mode, repo, bundle, cache, source):
    repo, bundle, cache, source = map(Path,(repo,bundle,cache,source))
    # Helper imports resolve only from the package extracted after ZIP verification.
    sys.path.insert(0,str(bundle))
    legacy = runpy.run_path(str(bundle/'recover_training_prerequisites.py'))
    from restore_ruod_images import restore_one
    from apply_ruod_visual_families import contract
    spec = json.loads((bundle/'recovery_manifest.json').read_text())
    spec_sha = digest(bundle/'recovery_manifest.json')
    cache.mkdir(parents=True,exist_ok=True)
    local = Path(tempfile.mkdtemp(prefix='ulgf-'+mode+'-',dir='/content'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    reports = repo/'_local_archive/runtime_reports'/stamp
    reports.mkdir(parents=True,exist_ok=False)
    report = dict(mode=mode,status='RUNNING',training_ready=False,training_started=False,
                  membership_changed=False,checks={},python='/content/ulgf-training-py37/bin/python')
    env = dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    for key in ('PYTHONPATH','PYTHONHOME'): env.pop(key,None)
    prefix = Path('/content/ulgf-training-py37'); python = prefix/'bin/python'

    def run(name,args,timeout=7200):
        print(name+' ...',flush=True)
        log = local/(name+'.log'); start = time.monotonic()
        try:
            with log.open('w',encoding='utf-8') as stream:
                p = subprocess.Popen(args,cwd=str(repo),env=env,stdout=stream,stderr=subprocess.STDOUT)
                try:
                    while p.poll() is None:
                        if time.monotonic()-start>timeout: raise TimeoutError(name)
                        try: p.wait(timeout=20)
                        except subprocess.TimeoutExpired: print(name+' running; '+str(int(time.monotonic()-start))+' seconds',flush=True)
                    if p.returncode: raise RuntimeError(name+': '+log.read_text(errors='replace')[-2200:])
                except BaseException:
                    if p.poll() is None:
                        p.terminate()
                        try: p.wait(timeout=10)
                        except subprocess.TimeoutExpired: p.kill(); p.wait()
                    raise
        finally:
            if log.exists(): shutil.copyfile(log,reports/log.name)
        return log.read_text(errors='replace')

    def install_python(installer):
        if digest(installer)!=legacy['INSTALLER_SHA']: raise ValueError('Untrusted installer')
        if prefix.exists() and not python.is_file(): raise ValueError('Partial Python prefix preserved; inspect '+str(prefix))
        if not python.is_file(): run('python_install',['bash',str(installer),'-b','-p',str(prefix)])
        run('python_version',[str(python),'-c','import sys; assert sys.version_info[:3]==(3,7,16),sys.version; print(sys.version)'])

    def verify_split(base):
        datasets = {}
        for split, expected in spec['base_contracts'].items():
            data = json.loads((base/'annotations'/('instances_'+split+'.json')).read_text())
            if contract(data)!=expected: raise ValueError('Split contract changed: '+split)
            datasets[split]=data
        if digest(base/'annotations/instances_test.json')!=spec['official_test_sha256']: raise ValueError('Official test changed')
        paths=json.loads((base/'paths.json').read_text())
        if paths['image_prefix']!='/content/ulgf-ruod-derived-v1/images': raise ValueError('Unexpected image root')
        return datasets

    try:
        if mode=='cpu':
            # No GPU detection, CUDA allocation, torch import, or trainer execution.
            if (cache/'READY.json').exists():
                check_ready(cache,spec_sha)
                report['status']='CPU_PREPARATION_PASS'
                report['cache_reused']=True
                report['next']='Existing complete cache verified. Switch to GPU and run the GPU cell.'
                return report
            candidates=[repo/'outputs/ruod-visual-families-v11-expanded30/split',
                repo/'_local_archive/20260930_github_cleanup/outputs/ruod-visual-families-v11-expanded30/split',
                Path('/content/ulgf-ruod-split-v11-reviewed'),cache/'split']
            base=next((p for p in candidates if (p/'annotations/instances_train.json').is_file()),None)
            if base is None: raise FileNotFoundError('Verified v11 split backup missing on Drive')
            datasets=verify_split(base)
            for origin,expected in spec['source_annotation_hashes'].items():
                if digest(source/'RUOD_ANN'/('instances_'+origin+'.json'))!=expected: raise ValueError('Source annotation hash mismatch')
            for name in ['paths.json','summary.json']+['annotations/instances_'+s+'.json' for s in datasets]:
                publish_file(base/name,cache/'split'/name)
            jobs=[]
            for split in ('train','validation','test'):
                origin='test' if split=='test' else 'train'
                for item in datasets[split]['images']:
                    record=spec['records'][origin][str(item['id'])]
                    jobs.append((item['file_name'],safe_target(source/'RUOD_pic'/origin,record['file_name']),record,item))
            jobs.sort(key=lambda row:row[0])
            if len({j[0] for j in jobs})!=len(jobs): raise ValueError('Duplicate destination')
            index={}; image_root=Path('/content/ulgf-ruod-derived-v1/images')
            for start in range(0,len(jobs),500):
                group=jobs[start:start+500]; name='images-%03d.tar'%(start//500)
                meta=cache/(name+'.json'); target=cache/name
                if meta.exists() and target.exists():
                    record=json.loads(meta.read_text())
                    if record.get('evidence_sha256')!=spec_sha or set(record['files'])!={j[0] for j in group} or digest(target)!=record['sha256']:
                        raise ValueError('Cached shard invalid; preserved for inspection: '+name)
                    print('Reusing verified '+name,flush=True)
                else:
                    for i,(relative,original,evidence,item) in enumerate(group,1):
                        restore_one(original,safe_target(image_root,relative),evidence,item)
                        if i%50==0: print('Audited images: %d/%d'%(start+i,len(jobs)),flush=True)
                    record=make_shard(image_root,[j[0] for j in group],local/name)
                    record['evidence_sha256']=spec_sha
                    publish_file(local/name,target)
                    atomic_json(meta,record)
                index[name]=record
            atomic_json(cache/'image_index.json',index)
            report['checks']['audited_image_cache']=dict(status='PASS',images=len(jobs),shards=len(index))
            installer=cache/'miniconda37.sh'
            if not installer.exists():
                download=local/'miniconda37.sh'
                with urlopen(legacy['INSTALLER'],timeout=120) as src,download.open('wb') as dst: shutil.copyfileobj(src,dst)
                if digest(download)!=legacy['INSTALLER_SHA']: raise ValueError('Installer hash mismatch')
                publish_file(download,installer)
            install_python(installer)
            pins=legacy['TORCH']+legacy['DETECTION']+legacy['TRAINING']+['pip==22.3.1','setuptools==65.6.3','wheel==0.38.4']
            if (cache/'wheel_index.json').exists() and (cache/'wheels.tar').exists():
                wheels_record=json.loads((cache/'wheel_index.json').read_text())
                if wheels_record['pins']!=pins or wheels_record['sha256']!=digest(cache/'wheels.tar'):
                    raise ValueError('Existing wheel cache differs; preserved for inspection')
                print('Reusing verified dependency wheel cache',flush=True)
            else:
                wheels=local/'wheels'; wheels.mkdir()
                # Resolve/download on CPU. GPU installation is offline.
                run('wheel_download',[str(python),'-m','pip','download','--only-binary=:all:',
                    '--dest',str(wheels),'--extra-index-url',legacy['TORCH_INDEX'],'--find-links',legacy['MMCV_INDEX']]+pins)
                wheels_record={p.name:dict(sha256=digest(p),size=p.stat().st_size) for p in wheels.glob('*.whl')}
                if not wheels_record: raise ValueError('No wheels downloaded')
                make_shard(wheels,sorted(wheels_record),local/'wheels.tar')
                publish_file(local/'wheels.tar',cache/'wheels.tar')
                atomic_json(cache/'wheel_index.json',dict(sha256=digest(cache/'wheels.tar'),files=wheels_record,pins=pins))
            names=list(index)+['image_index.json','wheel_index.json','wheels.tar','miniconda37.sh']
            names+=['split/'+n for n in ['paths.json','summary.json']+['annotations/instances_'+s+'.json' for s in datasets]]
            ready=dict(status='CPU_PREPARATION_PASS',evidence_sha256=spec_sha,
                training_ready=False,images=len(jobs),files={n:digest(cache/n) for n in names})
            atomic_json(cache/'READY.json',ready)
            check_ready(cache,spec_sha)
            report['status']='CPU_PREPARATION_PASS'
            report['next']='Switch to GPU only now, then run the GPU cell. No source-image recovery on GPU.'
        else:
            check_ready(cache,spec_sha)
            run('gpu_driver',['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],120)
            data=verify_split(cache/'split')
            expected={i['file_name'] for d in data.values() for i in d['images']}
            index=json.loads((cache/'image_index.json').read_text())
            flat=[n for record in index.values() for n in record['files']]
            if len(flat)!=len(set(flat)) or set(flat)!=expected: raise ValueError('Cache coverage differs from split')
            required=sum(info['size'] for record in index.values() for info in record['files'].values())
            required+=3*(cache/'wheels.tar').stat().st_size+8*1024**3
            if shutil.disk_usage(local).free<required: raise OSError('Insufficient local disk for verified extraction and environment')
            image_root=Path('/content/ulgf-ruod-derived-v1/images')
            for name,record in index.items():
                temp=local/name; shutil.copyfile(safe_target(cache,name),temp)
                restore_shard(temp,record,image_root)
                print('Restored '+name,flush=True)
            base=Path('/content/ulgf-ruod-split-v11-reviewed')
            if base.exists(): verify_split(base)
            else:
                staging=local/'split'; shutil.copytree(cache/'split',staging); verify_split(staging); staging.rename(base)
            env['RUOD_SPLIT_ROOT']=str(base)
            report['checks']['cached_images_and_split']=dict(status='PASS',images=len(flat))
            installer=local/'miniconda37.sh'; shutil.copyfile(cache/'miniconda37.sh',installer)
            install_python(installer)
            wheel_index=json.loads((cache/'wheel_index.json').read_text())
            shutil.copyfile(cache/'wheels.tar',local/'wheels.tar')
            restore_shard(local/'wheels.tar',wheel_index,local/'wheels')
            pip=[str(python),'-m','pip','install','--no-index','--find-links',str(local/'wheels')]
            run('offline_environment_install',pip+wheel_index['pins'])
            run('pip_check',[str(python),'-m','pip','check'])
            report['checks']['offline_dependencies']=dict(status='PASS')
            probe=runpy.run_path(str(bundle/'colab_training_preflight.py'))['PROBE']
            result=json.loads(run('cuda_ops_and_trainer',[str(python),'-c',probe]).strip().splitlines()[-1])
            if result.get('compiled_ops')!='CPU_AND_CUDA_PASS' or result.get('trainer_import')!='PASS': raise ValueError('Incomplete operator qualification')
            report['checks']['cuda_ops_and_trainer']=dict(status='PASS',details=result)
            for pattern in ('test_training_contract.py','test_training_runtime.py'):
                code="import unittest,sys; s=unittest.defaultTestLoader.discover('tests',pattern=sys.argv[1]); r=unittest.TextTestRunner(verbosity=2).run(s); sys.exit(0 if r.wasSuccessful() and r.testsRun and not r.skipped else 1)"
                run(pattern,[str(python),'-c',code,pattern])
                report['checks'][pattern]=dict(status='PASS',skips_allowed=False)
            run('dataset_paths',[str(python),'-c',
                "import os; from mmcv import Config; from ulgf_baseline.reviewed_data import reviewed_paths; c=Config.fromfile('configs/data/ruod_256x256.py'); p=reviewed_paths(os.environ['RUOD_SPLIT_ROOT']); assert c.data.train.ann_file==p['train']; assert c.data.val.ann_file==p['validation']; print('PASS')"])
            report['checks']['dataset_paths']=dict(status='PASS',scope='Manifest wiring only; not full transforms or model input validation')
            run('pip_freeze',[str(python),'-m','pip','freeze'])
            report['status']='GPU_ENVIRONMENT_QUALIFICATION_PASS'
            report['remaining']=['Leakage clearance','Approved initialization and configuration','Real-model update and resume','Overfit pilot and visual acceptance']
    except BaseException as error:
        report['status']='FAILED';report['error']=str(error) or type(error).__name__
        raise
    finally:
        atomic_json(reports/'report.json',report)
        print('FINAL REPORT: '+str(reports/'report.json'),flush=True)
        print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['cpu','gpu'])
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--source',type=Path,default='/content/drive/MyDrive/ULGF-assets/datasets/RUOD')
    args=parser.parse_args();main(args.mode,args.repo,args.bundle,args.cache,args.source)
