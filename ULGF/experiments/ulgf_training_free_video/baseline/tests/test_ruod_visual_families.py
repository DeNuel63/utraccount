import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from tools.build_ruod_visual_manifest import reconstruct_export_image
from tools.apply_ruod_visual_families import families_from_edges, contract, revise, apply, write, digest_file, read


class VisualFamilyTests(unittest.TestCase):
    def test_reconstruction_respects_format_and_mode_not_input_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'misleading.jpg'
            for mode,fmt,expected in [('RGB','JPEG','7.jpg'),('RGB','PNG','7.png'),
                                      ('RGBA','PNG','7.png'),('CMYK','JPEG','7.png')]:
                with self.subTest(mode=mode,fmt=fmt):
                    Image.new(mode,(10,12)).save(path,format=fmt)
                    item=dict(id=7,file_name=path.name,width=10,height=12)
                    record=dict(stored_size=[10,12],source_mode=mode)
                    self.assertEqual(reconstruct_export_image(item,record,path)['file_name'],expected)

    def fixture(self):
        datasets = {}
        for split, ids in [('train',[1,4,10]),('validation',[2,5,20]),('test',[3,30])]:
            datasets[split] = dict(categories=[dict(id=1,name='fish')],
                images=[dict(id=i,file_name=('test/' if split=='test' else 'train/')+str(i)+'.jpg',width=10,height=10) for i in ids],
                annotations=[dict(id=i,image_id=i,category_id=1,bbox=[0,0,2,2]) for i in ids])
        edges = [dict(left=a,right=b,review_status='visually_reviewed') for a,b in
                 [('train:1','validation:2'),('validation:2','test:3'),('train:4','validation:5')]]
        groups = families_from_edges(edges)
        manifest = dict(base_contracts={s:contract(d) for s,d in datasets.items()},reviewed_edges=edges,
                        families=[dict(members=g) for g in groups],
                        members={key:{} for g in groups for key in g})
        return datasets, manifest

    def test_transitive_exclusion_and_non_test_family_move(self):
        datasets, manifest = self.fixture()
        result, actions, report = revise(datasets, manifest)
        self.assertEqual([i['id'] for i in result['train']['images']], [10])
        self.assertEqual([i['id'] for i in result['validation']['images']], [4,5,20])
        self.assertEqual(result['test'],datasets['test'])
        self.assertEqual(report['excluded_images'],2)
        self.assertEqual(report['moved_images'],1)
        self.assertEqual(report['unreviewed_candidates_applied'],0)

    def test_unreviewed_edge_and_changed_membership_rejected(self):
        datasets, manifest = self.fixture()
        manifest['reviewed_edges'][0]['review_status']='candidate'
        with self.assertRaises(ValueError):
            revise(datasets,manifest)
        datasets, manifest = self.fixture()
        datasets['train']['images'][0]['width']=11
        with self.assertRaises(ValueError):
            revise(datasets,manifest)

    def test_hash_gate_and_test_manifest_byte_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,base,output,images=[root/name for name in ('source','v1','v2','images')]
            datasets,manifest=self.fixture()
            for split,data in datasets.items():
                write(base/'annotations'/('instances_'+split+'.json'),data)
                for item in data['images']:
                    path=images/item['file_name']
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(b'derived-image')
            write(base/'paths.json',dict(image_prefix=str(images)))
            manifest['source_annotation_hashes']={}
            for origin in ('train','test'):
                path=source/'RUOD_ANN'/('instances_'+origin+'.json')
                write(path,dict(example=origin))
                manifest['source_annotation_hashes'][origin]=digest_file(path)
            for key in manifest['members']:
                split,image_id=key.split(':')
                origin='test' if split=='test' else 'train'
                path=source/'RUOD_pic'/origin/(image_id+'.jpg')
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(key.encode())
                manifest['members'][key]=dict(origin=origin,file_name=path.name,source_sha256=digest_file(path))
            manifest_path=root/'manifest.json'
            write(manifest_path,manifest)
            changed=source/'RUOD_pic/train/1.jpg'
            original=changed.read_bytes()
            changed.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                apply(source,base,output,manifest_path)
            self.assertFalse(output.exists())
            changed.write_bytes(original)
            before=digest_file(base/'annotations/instances_test.json')
            with contextlib.redirect_stdout(io.StringIO()):
                apply(source,base,output,manifest_path)
            self.assertEqual(digest_file(output/'annotations/instances_test.json'),before)
            self.assertEqual(digest_file(base/'annotations/instances_test.json'),before)
            self.assertEqual(read(output/'summary.json')['original_file_hashes'],'PASS')
            with self.assertRaises(ValueError):
                apply(source,base,output,manifest_path)


if __name__=='__main__':
    unittest.main()
