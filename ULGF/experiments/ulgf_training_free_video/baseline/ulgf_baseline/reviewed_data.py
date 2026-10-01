"""Resolve a reviewed split without ever using official test for validation."""
import hashlib
import json
from pathlib import Path

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def reviewed_paths(root):
    root=Path(root).resolve()
    paths=json.loads((root/'paths.json').read_text())
    prefix=Path(paths['image_prefix']).resolve()
    result={'image_prefix':str(prefix)}
    ids=[]
    for split in ('train','validation'):
        path=root/'annotations'/('instances_'+split+'.json')
        data=json.loads(path.read_text())
        if not data['images']:
            raise ValueError('Empty '+split)
        for item in data['images']:
            target=(prefix/item['file_name']).resolve()
            target.relative_to(prefix)
            if not target.is_file():
                raise FileNotFoundError(target)
        result[split]=str(path)
        result[split+'_sha256']=sha256(path)
        ids.append({i['id'] for i in data['images']})
    if ids[0]&ids[1]:
        raise ValueError('Train and validation image IDs overlap')
    if result['train_sha256']==result['validation_sha256']:
        raise ValueError('Train and validation manifests identical')
    test=root/'annotations/instances_test.json'
    if test.exists() and result['validation_sha256']==sha256(test):
        raise ValueError('Official test cannot serve as validation')
    return result
