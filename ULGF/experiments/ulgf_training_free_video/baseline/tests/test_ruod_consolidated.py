import copy
import json
import tempfile
import unittest
from pathlib import Path
from tools.apply_ruod_consolidated import revise, apply
from tools.apply_ruod_historical22 import contract, digest

def fixture():
    def dataset(ids):
        return dict(images=[dict(id=i,file_name=str(i)+'.jpg') for i in ids],
                    annotations=[dict(id=i,image_id=i,category_id=1) for i in ids],categories=[dict(id=1,name='fish')])
    data=dict(train=dataset([1,2,3,4]),validation=dataset([5,6]),test=dataset([7]))
    members={('test' if i==7 else 'train')+':'+str(i):dict(origin='test' if i==7 else 'train',id=i,file_name=str(i)+'.jpg',source_sha256='') for i in range(1,9)}
    edges=[('train:1','train:8'),('train:8','test:7'),('train:5','test:7'),('train:2','train:6')]
    m=dict(members=members,base_contracts={s:contract(d) for s,d in data.items()},
           reviewed_edges=[dict(left=a,right=b,review_status='visually_reviewed') for a,b in edges],
           proposed_actions=[dict(member=k,from_split=s,to_split=t) for k,s,t in [('train:1','train','exclude'),('train:5','validation','exclude'),('train:2','train','validation')]],
           expected_images=dict(train=2,validation=2,test=1))
    return data,m

class ConsolidatedTests(unittest.TestCase):
    def test_transitive_exclusions_and_move_preserve_annotations(self):
        data,m=fixture(); result,actions=revise(data,m)
        self.assertEqual([i['id'] for i in result['validation']['images']],[2,6])
        self.assertEqual([a['image_id'] for a in result['validation']['annotations']],[2,6])
        self.assertEqual(result['test'],data['test'])
        self.assertNotIn(8,[i['id'] for d in result.values() for i in d['images']])
    def test_exact_action_guard(self):
        data,m=fixture(); m['proposed_actions'].pop()
        with self.assertRaises(ValueError): revise(data,m)
    def test_unreviewed_edge_rejected(self):
        data,m=fixture(); m['reviewed_edges'][0]['review_status']='candidate'
        with self.assertRaises(ValueError): revise(data,m)
    def test_changed_base_rejected(self):
        data,m=fixture(); data['train']['images'][0]['file_name']='changed'
        with self.assertRaises(ValueError): revise(data,m)
    def test_apply_hash_failure_then_byte_preserving_success(self):
        data,m=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; base=root/'base'; images=root/'images'
            (base/'annotations').mkdir(parents=True); (source/'RUOD_ANN').mkdir(parents=True); images.mkdir()
            for s,d in data.items():
                (base/'annotations'/('instances_'+s+'.json')).write_text(json.dumps(d))
                for i in d['images']: (images/i['file_name']).write_bytes(b'fixture')
            (base/'paths.json').write_text(json.dumps(dict(image_prefix=str(images))))
            m['source_annotation_hashes']={}
            for s in ('train','test'):
                p=source/'RUOD_ANN'/('instances_'+s+'.json'); p.write_text('{}')
                m['source_annotation_hashes'][s]=digest(p)
            for r in m['members'].values():
                p=source/'RUOD_pic'/r['origin']/r['file_name']; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'original')
                r['source_sha256']=digest(p)
            m['official_test_sha256']=digest(base/'annotations/instances_test.json')
            manifest=root/'manifest.json'; manifest.write_text(json.dumps(m))
            changed=source/'RUOD_pic/train/1.jpg'; changed.write_bytes(b'changed')
            with self.assertRaises(ValueError): apply(source,base,root/'out',manifest)
            changed.write_bytes(b'original')
            # Preservation mode refuses a proposal that modifies validation.
            m['preserve_validation']=True
            manifest.write_text(json.dumps(m))
            with self.assertRaises(ValueError): apply(source,base,root/'invalid-preserve',manifest)
            self.assertFalse((root/'invalid-preserve').exists())
            # Train-only exclusion keeps the original validation serialization.
            m['reviewed_edges']=m['reviewed_edges'][:2]
            m['proposed_actions']=m['proposed_actions'][:1]
            m['expected_images']=dict(train=3,validation=2,test=1)
            manifest.write_text(json.dumps(m))
            apply(source,base,root/'preserved',manifest)
            for split in ('validation','test'):
                name='instances_'+split+'.json'
                self.assertEqual((base/'annotations'/name).read_bytes(),(root/'preserved/annotations'/name).read_bytes())
            self.assertFalse((root/'out').exists())
            changed.write_bytes(b'original'); apply(source,base,root/'out',manifest)
            self.assertEqual((base/'annotations/instances_test.json').read_bytes(),(root/'out/annotations/instances_test.json').read_bytes())
            with self.assertRaises(ValueError): apply(source,base,root/'out',manifest)

if __name__=='__main__': unittest.main()
