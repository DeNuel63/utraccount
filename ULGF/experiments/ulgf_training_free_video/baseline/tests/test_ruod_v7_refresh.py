import unittest
from tools.refresh_ruod_v7 import components


class V7RefreshTests(unittest.TestCase):
    def row(self,origin,i):
        return dict(origin=origin,id=i)

    def test_transitive_history_includes_excluded_anchor(self):
        a,b,c=self.row('train',1),self.row('train',2),self.row('test',1)
        groups=components([dict(left=a,right=b),dict(left=b,right=c)])
        self.assertEqual(groups,[dict(members=['test:1','train:1','train:2'],touches_test=True)])

    def test_source_namespaces_and_disconnected_components(self):
        edges=[dict(left=self.row('train',1),right=self.row('train',2)),
               dict(left=self.row('test',1),right=self.row('test',2))]
        self.assertEqual(len(components(edges)),2)
        self.assertEqual(sum(g['touches_test'] for g in components(edges)),1)

    def test_duplicate_edges_do_not_duplicate_members(self):
        e=dict(left=self.row('train',1),right=self.row('test',1))
        self.assertEqual(components([e,e]),components([e]))
        self.assertEqual(components([]),[])


if __name__=='__main__':
    unittest.main()
