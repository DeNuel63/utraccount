"""One invocation, one report. Fail closed; never launch full training.

Current v11 has confirmed unresolved leakage. Independent diagnostics may run,
but model updates/pilot are blocked until the data gate genuinely passes.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REQUIRED=('split_contracts','image_availability','leakage_clearance','training_contract_tests',
          'training_runtime_tests','dataset_wiring','initialization','training_configuration',
          'gpu','pinned_runtime','dependency_consistency','trainer_syntax','launcher_syntax',
          'real_update_and_resume','pilot_acceptance')
CONTRACTS={'train':'f499cba46d2809d2c23359e0eae89bff84aae2a3a8ea270352a57e3f107a4d37',
           'validation':'e38865afa3dccc1245f567676fcb46bce45c778104a202ed9c31b0177a064771',
           'test':'c5328dee390d0d9734aa5b8ec63fcce1ee9f72ef1e28c0536634f93a1a0bb20f'}
TEST_SHA='554b3a631fccf53f651aee8b8e0fcccf2c4a3763325d86ab9807449a7f4cff8f'

def ready(checks):
    return all(checks.get(k,{}).get('status')=='PASS' for k in REQUIRED)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def main(repo,python37,emit=True):
    repo=Path(repo); python37=Path(python37)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    fallback=Path('/content') if Path('/content').is_dir() else Path.cwd()
    local=fallback/('ulgf-final-preflight-'+stamp); local.mkdir()
    report=dict(recorded_utc=stamp,scope='v11 consolidated diagnostic preflight',checks={},
        training_ready=False,training_started=False,installs_performed=False,membership_changed=False)
    base=Path('/content/ulgf-ruod-split-v11-reviewed')
    if not base.is_dir(): base=repo/'outputs/ruod-visual-families-v11-expanded30/split'
    env=dict(os.environ,RUOD_SPLIT_ROOT=str(base),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')

    def save():
        report['training_ready']=ready(report['checks'])
        report['status']='READY' if report['training_ready'] else 'NOT_READY'
        (local/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

    def run(name,action):
        print(name+' ...',flush=True)
        try: report['checks'][name]=dict(status='PASS',details=action())
        except Exception as e: report['checks'][name]=dict(status='FAIL',error=str(e))
        save(); print(name+': '+report['checks'][name]['status'],flush=True)

    def block(name,reason):
        report['checks'][name]=dict(status='BLOCKED',reason=reason); save()
        print(name+': BLOCKED — '+reason,flush=True)

    def command(name,args,timeout=600):
        p=subprocess.run(args,cwd=str(repo),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
        (local/(name+'.log')).write_text(p.stdout,encoding='utf-8')
        if p.returncode: raise RuntimeError('Exit {}. See {}.log\n{}'.format(p.returncode,name,p.stdout[-2500:]))
        return p.stdout

    datasets={}
    def splits():
        for s,expected in CONTRACTS.items():
            d=json.loads((base/'annotations'/('instances_'+s+'.json')).read_text())
            payload={k:sorted(d[k],key=lambda r:r['id']) for k in ('images','annotations','categories')}
            if hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=expected:
                raise ValueError('Split differs from reviewed v11: '+s)
            datasets[s]=d
        if sha(base/'annotations/instances_test.json')!=TEST_SHA: raise ValueError('Official test bytes changed')
        return dict(base=str(base),counts={s:len(d['images']) for s,d in datasets.items()})

    def images():
        if report['checks']['split_contracts']['status']!='PASS': raise ValueError('Split checks failed')
        prefix=Path(json.loads((base/'paths.json').read_text())['image_prefix']).resolve()
        missing=[]
        for d in datasets.values():
            for i in d['images']:
                p=(prefix/i['file_name']).resolve(); p.relative_to(prefix)
                if not p.is_file(): missing.append(i['file_name'])
        if missing: raise ValueError('{} derived images missing; restore from audited recovery bundle. First: {}'.format(len(missing),missing[:3]))
        return dict(image_root=str(prefix),scope='Existence and path containment; prior pixel audit not repeated')

    def tests(pattern):
        code="""import unittest,sys,json
s=unittest.defaultTestLoader.discover('tests',pattern=sys.argv[1])
r=unittest.TextTestRunner(verbosity=2).run(s)
print(json.dumps(dict(run=r.testsRun,skipped=len(r.skipped))))
sys.exit(0 if r.wasSuccessful() and r.testsRun>0 and not r.skipped else 1)
"""
        return command(pattern,[str(python37),'-c',code,pattern])[-1000:]

    def environment():
        probe=runpy.run_path(str(repo/'tools/colab_training_preflight.py'))['PROBE']
        text=command('environment',[str(python37),'-c',probe])
        result=json.loads(text.strip().splitlines()[-1])
        if result.get('compiled_ops')!='CPU_AND_CUDA_PASS' or result.get('trainer_import')!='PASS':
            raise ValueError('Training packages/compiled ops incomplete: '+json.dumps(result))
        return result

    def wiring():
        code="""from mmcv import Config
from pathlib import Path
from ulgf_baseline.reviewed_data import reviewed_paths
import os
c=Config.fromfile('configs/data/ruod_256x256.py')
r=reviewed_paths(os.environ['RUOD_SPLIT_ROOT'])
assert Path(c.data.train.ann_file).resolve()==Path(r['train']).resolve()
assert Path(c.data.val.ann_file).resolve()==Path(r['validation']).resolve()
assert Path(c.data.val.ann_file).name=='instances_validation.json'
assert c.data.train.ann_file!=c.data.val.ann_file
print('Train/validation configuration points to reviewed manifests; full transform integration remains gated by split approval.')
"""
        return command('dataset_wiring',[str(python37),'-c',code])

    try:
        run('split_contracts',splits); run('image_availability',images)
        block('leakage_clearance','V11 contains 11 visually supported unresolved cross-split links in the diagnostic review; 4,725 weaker flags remain. Zero score<=4 links is not clearance.')
        run('trainer_syntax',lambda:ast.parse((repo/'train_UWLGM.py').read_text()) and 'PASS in notebook interpreter; pinned import checked separately')
        run('launcher_syntax',lambda:command('launcher',['bash','-n','tools/dist_train.sh']))
        if python37.is_file():
            run('training_contract_tests',lambda:tests('test_training_contract.py'))
            run('training_runtime_tests',lambda:tests('test_training_runtime.py'))
            run('dependency_consistency',lambda:command('pip_check',[str(python37),'-m','pip','check']))
            run('dataset_wiring',wiring)
        else:
            for name in ('training_contract_tests','training_runtime_tests','dependency_consistency','dataset_wiring'):
                block(name,'Pinned Python 3.7 missing: '+str(python37))
        # Probe existing GPU only; this cell does not require switching to GPU
        # while the known leakage blocker remains unresolved.
        run('gpu',lambda:command('gpu',['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader']))
        if python37.is_file(): run('pinned_runtime',environment)
        else: block('pinned_runtime','Pinned Python missing; environment installation not performed.')
        report['checkpoint_candidates']={str(p):p.is_dir() for p in
            [Path('/content/ulgf-official-download/ruod_256_bz16/checkpoint'),repo/'assets/checkpoint/final']}
        block('initialization','Training initialization not approved and fully verified. Existing RUOD-trained weights may already have seen held-out images; directory presence is not verification.')
        block('training_configuration','Final split and initialization must be approved before freezing model/optimizer/prompt/loss settings.')
        block('real_update_and_resume','Requires cleared data, locked initialization/configuration and validated GPU environment. Toy resume tests are not a real-model resume test.')
        block('pilot_acceptance','Requires real update/resume PASS and a bounded pilot with validation and generated-image review. Not executed.')
    except Exception as error:
        report['unexpected_error']=str(error)
    finally:
        for name in REQUIRED:
            report['checks'].setdefault(name,dict(status='BLOCKED',reason='Earlier prerequisite failed or execution interrupted.'))
        report['blocking_gates']=[k for k in REQUIRED if report['checks'][k]['status']!='PASS']
        save()
        destination=repo/'outputs/consolidated-preflight'/stamp
        try:
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(local,destination)
            if sha(local/'report.json')!=sha(destination/'report.json'): raise ValueError('Report readback mismatch')
            if emit: print('FINAL REPORT:',destination/'report.json')
        except Exception as e:
            print('Drive report copy failed:',e,'Local report:',local/'report.json')
        if emit:
            print(json.dumps(report,indent=2))
            print('No full training, pilot, installation or membership changes performed.')
    return report

if __name__=='__main__':
    main('/content/drive/MyDrive/ULGF-main','/content/miniconda37/bin/python')
