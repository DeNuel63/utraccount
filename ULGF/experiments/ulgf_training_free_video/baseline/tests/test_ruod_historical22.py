import copy
import unittest
import json
import tempfile
from pathlib import Path
from tools.apply_ruod_historical22 import contract, revise, apply, digest


class Historical22Tests(unittest.TestCase):
    def fixture(self):
        def dataset(ids):
            return dict(images=[dict(id=i,file_name=str(i)+'.jpg') for i in ids],
                annotations=[dict(id=i,image_id=i,category_id=1) for i in ids],
                categories=[dict(id=1,name='fish')])
        data = dict(train=dataset(range(1,24)),validation=dataset([100]),test=dataset([200]))
        members = {'train:'+str(i):dict(origin='train',id=i) for i in range(1,23)}
        members.update({'train:99':dict(origin='train',id=99),'test:200':dict(origin='test',id=200)})
        edges = [dict(left='train:'+str(i),right='train:99',review_status='visually_reviewed') for i in range(1,23)]
        edges.append(dict(left='train:99',right='test:200',review_status='visually_reviewed'))
        manifest = dict(base_contracts={s:contract(d) for s,d in data.items()},members=members,
            reviewed_edges=edges,proposed_exclusions=['train:'+str(i) for i in range(1,23)])
        return data, manifest

    def test_transitive_excluded_anchor_not_reintroduced(self):
        data, manifest = self.fixture()
        original = copy.deepcopy(data)
        result = revise(data, manifest)
        self.assertEqual([i['id'] for i in result['train']['images']],[23])
        self.assertEqual(result['test'],data['test'])
        self.assertEqual(result['validation'],data['validation'])
        self.assertEqual(data,original)

    def test_no_test_connection_rejected(self):
        data, manifest = self.fixture()
        manifest['reviewed_edges'].pop()
        with self.assertRaisesRegex(ValueError,'No explicit reviewed path'):
            revise(data,manifest)

    def test_unreviewed_link_rejected(self):
        data, manifest = self.fixture()
        manifest['reviewed_edges'][0]['review_status'] = 'candidate'
        with self.assertRaisesRegex(ValueError,'Unreviewed'):
            revise(data,manifest)

    def test_changed_base_rejected(self):
        data, manifest = self.fixture()
        data['train']['images'].pop()
        with self.assertRaisesRegex(ValueError,'Base contract mismatch'):
            revise(data,manifest)

    def test_duplicate_exclusions_rejected(self):
        data, manifest = self.fixture()
        manifest['proposed_exclusions'][-1] = manifest['proposed_exclusions'][0]
        with self.assertRaisesRegex(ValueError,'22 unique'):
            revise(data,manifest)

    def test_fourteen_application_preserves_bytes_and_rejects_changed_hash(self):
        data,manifest=self.fixture()
        manifest['proposed_exclusions']=manifest['proposed_exclusions'][:14]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; base=root/'base'; images=root/'images'
            (base/'annotations').mkdir(parents=True)
            (source/'RUOD_ANN').mkdir(parents=True)
            images.mkdir()
            manifest['source_annotation_hashes']={}
            for split,d in data.items():
                path=base/'annotations'/('instances_'+split+'.json')
                path.write_text(json.dumps(d,indent=3))
                for i in d['images']:
                    (images/i['file_name']).write_bytes(b'derived-fixture')
                if split!='validation':
                    original=source/'RUOD_ANN'/path.name
                    original.write_bytes(path.read_bytes())
                    manifest['source_annotation_hashes'][split]=digest(original)
            for member in manifest['members'].values():
                member['file_name']=str(member['id'])+'.jpg'
                path=source/'RUOD_pic'/member['origin']/member['file_name']
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'source-fixture')
                member['source_sha256']=digest(path)
            (base/'paths.json').write_text(json.dumps(dict(image_prefix=str(images))))
            manifest['official_test_sha256']=digest(base/'annotations/instances_test.json')
            m=root/'manifest.json'; m.write_text(json.dumps(manifest))
            out=root/'v8'
            apply(source,base,out,m,14)
            self.assertEqual(len(json.loads((out/'annotations/instances_train.json').read_text())['images']),9)
            for split in ('validation','test'):
                name='instances_'+split+'.json'
                self.assertEqual((base/'annotations'/name).read_bytes(),(out/'annotations'/name).read_bytes())
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Source image hash mismatch'):
                apply(source,base,root/'rejected',m,14)
            self.assertFalse((root/'rejected').exists())


if __name__ == '__main__':
    unittest.main()
