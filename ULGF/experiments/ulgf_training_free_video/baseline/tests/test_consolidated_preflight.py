import unittest
from tools.consolidated_preflight import ready,REQUIRED

class PreflightGateTests(unittest.TestCase):
    def test_empty_report_never_ready(self):
        self.assertFalse(ready({}))
    def test_all_required_checks_must_pass(self):
        self.assertTrue(ready({k:dict(status='PASS') for k in REQUIRED}))
    def test_blocked_failed_skipped_missing_never_ready(self):
        for gate in REQUIRED:
            for state in ('FAIL','BLOCKED','SKIPPED','NOT_RUN'):
                checks={k:dict(status='PASS') for k in REQUIRED}; checks[gate]['status']=state
                self.assertFalse(ready(checks),(gate,state))
            checks={k:dict(status='PASS') for k in REQUIRED if k!=gate}
            self.assertFalse(ready(checks))
    def test_toy_resume_cannot_clear_real_model_gate(self):
        checks={k:dict(status='PASS') for k in REQUIRED}
        checks['real_update_and_resume']=dict(status='BLOCKED')
        self.assertFalse(ready(checks))

if __name__=='__main__': unittest.main()
