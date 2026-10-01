import ast
import hashlib
import json
from pathlib import Path
import sys
import os
import unittest

ROOT = Path(os.environ['ULGF_EXPERIMENT_ROOT']) if 'ULGF_EXPERIMENT_ROOT' in os.environ else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scheduler_control import corrected_source, TARGET

class ScheduleControlTests(unittest.TestCase):
    def source(self):
        return (ROOT/'baseline/pipeline_custom/pipeline_prior.py').read_text(encoding='utf-8')

    def branch(self, prior):
        tree = ast.parse(corrected_source(self.source()))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Name) and n.test.id == 'latents_prior_flag')
        scope = dict(latents_prior_flag=prior, timesteps_tensor=list(range(101)))
        fragment = ast.parse('')
        fragment.body = [node]
        exec(compile(fragment, '<branch-test>', 'exec'), scope)
        return scope['timesteps_tensor']

    def test_no_prior_retains_full_schedule(self):
        self.assertEqual(self.branch(False), list(range(101)))

    def test_prior_branch_unchanged(self):
        self.assertEqual(self.branch(True), list(range(5,101)))

    def test_only_slice_guard_changes(self):
        source = self.source()
        from scheduler_control import REPLACEMENT
        self.assertEqual(corrected_source(source).replace(REPLACEMENT, TARGET), source)

    def test_unknown_or_already_patched_source_rejected(self):
        for source in ('pass', self.source()+ '\n'+TARGET, corrected_source(self.source())):
            with self.assertRaises(ValueError): corrected_source(source)

    def test_frozen_snapshot_intact(self):
        manifest=json.loads((ROOT/'baseline_snapshot.json').read_text(encoding='utf-8'))
        for name, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'baseline'/name).read_bytes()).hexdigest(),expected,name)

if __name__ == '__main__': unittest.main()
