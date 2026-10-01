"""Exercise the actual Colab publication helpers without importing Colab."""
import ast
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path


class RecoveryPublicationTests(unittest.TestCase):
    def setUp(self):
        cell=Path(__file__).resolve().parents[1]/'colab/ULGF_checkpoint02_consolidated_cell.py'
        tree=ast.parse(cell.read_text(encoding='utf-8'))
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('tree_hashes','publish_verified')]
        namespace=dict(hashlib=hashlib,shutil=shutil)
        exec(compile(tree,str(cell),'exec'),namespace)
        self.publish=namespace['publish_verified']

    def test_partial_preserved_and_complete_copy_reusable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); candidate=root/'candidate'; destination=root/'split'
            candidate.mkdir(); (candidate/'summary.json').write_text('complete')
            (candidate/'annotations').mkdir(); (candidate/'annotations/test.json').write_bytes(b'official')
            destination.mkdir(); (destination/'summary.json').write_text('truncated')
            self.publish(candidate,destination)
            backups=list(root.glob('split.interrupted-*'))
            self.assertEqual(len(backups),1)
            self.assertEqual((backups[0]/'summary.json').read_text(),'truncated')
            self.assertEqual((destination/'annotations/test.json').read_bytes(),b'official')
            self.publish(candidate,destination)
            self.assertEqual(len(list(root.glob('split.interrupted-*'))),1)

    def test_absent_destination_and_extra_file_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); candidate=root/'candidate'; destination=root/'split'
            candidate.mkdir(); (candidate/'data').write_bytes(b'correct')
            self.publish(candidate,destination)
            (destination/'unexpected').write_bytes(b'keep me')
            self.publish(candidate,destination)
            backup=next(root.glob('split.interrupted-*'))
            self.assertEqual((backup/'unexpected').read_bytes(),b'keep me')

if __name__=='__main__': unittest.main()
