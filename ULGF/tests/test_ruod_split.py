import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from PIL import Image
from tools.audit_ruod_dataset import audit, file_hash
from tools.prepare_ruod_split import partition, prepare, read


class SplitTests(unittest.TestCase):
    def test_incremental_restoration_and_idempotent_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, derived, output = [Path(tmp) / name for name in ('source', 'derived', 'split')]
            approvals = {}
            for split, count in (('train', 8), ('test', 2)):
                directory = source / 'RUOD_pic' / split
                directory.mkdir(parents=True)
                data = dict(images=[], annotations=[], categories=[dict(id=1, name='fish')])
                for i in range(1, count+1):
                    exif = Image.Exif()
                    exif[274] = 0
                    path = directory / (str(i)+'.jpg')
                    color = (i*25, 0 if split == 'train' else 255, 0)
                    Image.new('RGB', (10, 10), color).save(path, exif=exif)
                    approvals[split+':'+str(i)] = file_hash(path)
                    data['images'].append(dict(id=i, file_name=path.name, width=10, height=10))
                    data['annotations'].append(dict(id=i, image_id=i, category_id=1, bbox=[0, 0, 5, 5]))
                ann = source / 'RUOD_ANN' / ('instances_'+split+'.json')
                ann.parent.mkdir(exist_ok=True)
                ann.write_text(json.dumps(data))
            with contextlib.redirect_stdout(io.StringIO()):
                audit(source, derived, True)
                wrong_approvals = dict(approvals)
                wrong_approvals['train:1'] = 'unreviewed'
                with self.assertRaises(ValueError):
                    prepare(source, derived, wrong_approvals, output)
                self.assertFalse((derived / 'images/train/1.jpg').exists())
                prepare(source, derived, approvals, output)
                first = read(output / 'annotations/instances_train.json')
                prepare(source, derived, approvals, output)
            self.assertEqual(first, read(output / 'annotations/instances_train.json'))
            self.assertEqual(read(derived / 'summary.json')['quarantined_images'], 0)
            self.assertEqual(len(read(derived / 'annotations/instances_test.json')['images']), 2)
            self.assertEqual(read(output / 'summary.json')['images'], dict(train=7, validation=1, official_test=2))
            for key, digest in approvals.items():
                split, image_id = key.split(':')
                self.assertEqual(file_hash(source / 'RUOD_pic' / split / (image_id+'.jpg')), digest)
                with Image.open(derived / 'images' / split / (image_id+'.jpg')) as image:
                    self.assertFalse(image.getexif())

    def fixture(self):
        categories = [dict(id=1, name='fish')]
        train = dict(categories=categories, images=[], annotations=[])
        test = dict(categories=categories, images=[dict(id=1, file_name='1.jpg', width=10, height=10)],
                    annotations=[dict(id=99, image_id=1, category_id=1, bbox=[0, 0, 2, 2])])
        records = dict(train=[], test=[])
        for i in range(1, 21):
            train['images'].append(dict(id=i, file_name=str(i)+'.jpg', width=10, height=10))
            train['annotations'].append(dict(id=i, image_id=i, category_id=1, bbox=[0, 0, 2, 2]))
            records['train'].append(dict(id=i, file_sha256=str(i), stored_pixels_sha256=str(i),
                                         display_pixels_sha256=str(i)))
        records['test'].append(dict(id=1, file_sha256='test', stored_pixels_sha256='1', display_pixels_sha256='1'))
        # Equal pixels, agreeing annotations: keep 2, drop 3.
        records['train'][2]['stored_pixels_sha256'] = '2'
        # Equal pixels, conflicting annotations: exclude both 4 and 5.
        records['train'][4]['stored_pixels_sha256'] = '4'
        train['annotations'][4]['bbox'] = [1, 1, 2, 2]
        return train, test, records

    def test_exclusions_reproducibility_and_test_preservation(self):
        inputs = self.fixture()
        result = partition(*inputs)
        self.assertEqual(result, partition(*inputs))
        train, val, test, exclusions, groups, report = result
        self.assertEqual(report['exclusion_counts'], dict(exact_match_to_official_test=1,
            redundant_identical_annotation_copy=1, conflicting_duplicate_annotations=2))
        ids = {x['id'] for x in train['images']} | {x['id'] for x in val['images']}
        self.assertEqual(ids, set(range(1, 21)) - {1, 3, 4, 5})
        self.assertEqual(test['annotations'], inputs[1]['annotations'])
        self.assertTrue(all(i['file_name'].startswith('train/') for i in train['images']))
        self.assertEqual(report['exact_hash_leakage'], 'PASS')

    def test_transitive_duplicate_connection_to_test(self):
        train, test, records = self.fixture()
        records['train'][1]['display_pixels_sha256'] = '1'
        result = partition(train, test, records)
        self.assertEqual(result[-1]['exclusion_counts']['exact_match_to_official_test'], 3)

    def test_category_mismatch_rejected(self):
        train, test, records = self.fixture()
        test['categories'] = [dict(id=1, name='diver')]
        with self.assertRaises(ValueError):
            partition(train, test, records)


if __name__ == '__main__':
    unittest.main()
