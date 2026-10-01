import ast
import copy
import hashlib
import json
import unittest
import zipfile
from pathlib import Path
from tools.apply_ruod_consolidated import revise

class Expanded30Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo=Path(__file__).resolve().parents[1]
        cls.archive=cls.repo/'colab/ULGF_checkpoint02_expanded30.zip'
        with zipfile.ZipFile(cls.archive) as z:
            cls.manifest=json.loads(z.read('reviewed_manifest.json'))
        cls.data={s:json.loads((cls.repo/'outputs/ruod-v10-refresh-base/annotations'/('instances_'+s+'.json')).read_text()) for s in ('train','validation','test')}

    def test_exact_partition_exclusions_and_annotations(self):
        original=copy.deepcopy(self.data)
        result,actions=revise(self.data,self.manifest)
        self.assertEqual(self.data,original)
        self.assertEqual(result['test'],original['test'])
        for split,count in [('train',25),('validation',5)]:
            removed={int(a['member'].split(':')[1]) for a in actions if a['from_split']==split}
            self.assertEqual(len(removed),count)
            self.assertEqual({i['id'] for i in result[split]['images']},{i['id'] for i in original[split]['images']}-removed)
            self.assertEqual(result[split]['annotations'],sorted([a for a in original[split]['annotations'] if a['image_id'] not in removed],key=lambda a:a['id']))
        self.assertEqual(len(actions),30)

    def test_omitted_validation_action_rejected(self):
        m=copy.deepcopy(self.manifest)
        m['proposed_actions']=[a for a in m['proposed_actions'] if a['from_split']!='validation']
        with self.assertRaises(ValueError): revise(self.data,m)

    def test_zip_cell_and_recovery_binding(self):
        cell=(self.repo/'colab/ULGF_checkpoint02_expanded30_cell.py').read_text()
        ast.parse(cell)
        self.assertIn(hashlib.sha256(self.archive.read_bytes()).hexdigest(),cell)
        self.assertIn('/content/ulgf-ruod-split-v10-reviewed',cell)
        self.assertIn('/content/ulgf-ruod-split-v11-reviewed',cell)
        with zipfile.ZipFile(self.archive) as z:
            self.assertIsNone(z.testzip())
            for name in z.namelist():
                if name.endswith('.py'): ast.parse(z.read(name).decode())
            recovery=json.loads(z.read('recovery_manifest.json'))
        self.assertEqual(recovery['base_contracts'],self.manifest['base_contracts'])
        self.assertEqual(sum(len(d) for d in recovery['records'].values()),12587)

if __name__=='__main__': unittest.main()
