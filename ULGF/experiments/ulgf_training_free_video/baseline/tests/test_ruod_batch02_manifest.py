import unittest
from tools.build_ruod_batch02_manifest import assemble_manifest
from tools.apply_ruod_visual_families import revise


class Batch02Tests(unittest.TestCase):
    def fixture(self):
        base={}
        for split,ids in [('train',[1,10]),('validation',[2,20]),('test',[3,30])]:
            base[split]=dict(categories=[dict(id=1,name='fish')],
                images=[dict(id=i,file_name=str(i)+'.jpg',width=10,height=10) for i in ids],
                annotations=[dict(id=i,image_id=i,category_id=1,bbox=[0,0,2,2]) for i in ids])
        members={}
        for old,split,i in [('train:1','train',1),('train:2','validation',2),('test:3','test',3)]:
            members[old]=dict(split=split,id=i,origin='test' if split=='test' else 'train',
                              file_name=str(i)+'.jpg',source_sha256='example')
        candidates=[dict(candidate_family='B02F01',shown_members=members,
            shown_edges=[dict(left='train:1',right='test:3'),dict(left='train:2',right='test:3')])]
        review=dict(batch_candidates_sha256='fixed',review_status='BATCH_VISUAL_REVIEW_COMPLETE',
            review_date='2026-09-23',reviews=[dict(candidate_family='B02F01',members_reviewed=3,
                links_reviewed=2,verdict='same_scene_family_supported',finding='Reviewed fixture')])
        return base,candidates,review

    def test_v2_moved_member_keys_and_transitive_exclusion(self):
        base,candidates,review=self.fixture()
        manifest=assemble_manifest(base,candidates,review,'fixed',{})
        self.assertIn('validation:2',manifest['members'])
        self.assertNotIn('train:2',manifest['members'])
        result,actions,report=revise(base,manifest)
        self.assertEqual(report['excluded_images'],2)
        self.assertEqual(report['moved_images'],0)
        self.assertEqual(result['test'],base['test'])

    def test_changed_review_inventory_and_incomplete_scope_fail(self):
        base,candidates,review=self.fixture()
        with self.assertRaises(ValueError):
            assemble_manifest(base,candidates,review,'changed',{})
        review['reviews'][0]['members_reviewed']=2
        with self.assertRaises(ValueError):
            assemble_manifest(base,candidates,review,'fixed',{})

    def test_unreviewed_candidate_component_not_imported(self):
        base,candidates,review=self.fixture()
        candidates.append(dict(candidate_family='unreviewed',shown_members={},shown_edges=[]))
        manifest=assemble_manifest(base,candidates,review,'fixed',{})
        self.assertEqual(len(manifest['reviewed_edges']),2)

    def test_batch03_metadata_uses_current_base_and_actual_edge_count(self):
        base,candidates,review=self.fixture()
        manifest=assemble_manifest(base,candidates,review,'fixed',{},batch_id='03',base_split='v3-reviewed')
        self.assertEqual(manifest['batch'],'03')
        self.assertEqual(manifest['base_split'],'v3-reviewed')
        self.assertIn('2 explicit batch-03',manifest['scope'])

    def test_training_family_requires_opt_in_and_moves_together(self):
        base,candidates,review=self.fixture()
        del candidates[0]['shown_members']['test:3']
        candidates[0]['shown_edges']=[dict(left='train:1',right='train:2')]
        review['reviews'][0].update(members_reviewed=2,links_reviewed=1)
        with self.assertRaisesRegex(ValueError,'non-test family'):
            assemble_manifest(base,candidates,review,'fixed',{})
        manifest=assemble_manifest(base,candidates,review,'fixed',{},batch_id='04',
                                   base_split='v4-reviewed',allow_training_families=True)
        self.assertEqual(manifest['families'][0]['action'],'keep_in_single_partition')
        result,actions,report=revise(base,manifest)
        self.assertEqual(report['excluded_images'],0)
        self.assertEqual(report['moved_images'],1)
        self.assertEqual(actions[0]['to_split'],'validation')
        self.assertEqual(result['test'],base['test'])
        self.assertEqual({i['id'] for i in result['validation']['images']},{1,2,20})

    def test_new_test_bridge_excludes_previously_grouped_validation_family(self):
        base,candidates,review=self.fixture()
        moved_image=base['train']['images'].pop(0)
        moved_ann=base['train']['annotations'].pop(0)
        base['validation']['images'].append(moved_image)
        base['validation']['annotations'].append(moved_ann)
        candidates[0]['shown_members']['train:1']['split']='validation'
        candidates[0]['shown_edges']=[dict(left='train:1',right='train:2'),
                                     dict(left='test:3',right='train:2')]
        manifest=assemble_manifest(base,candidates,review,'fixed',{},batch_id='05',base_split='v5-reviewed')
        result,actions,report=revise(base,manifest)
        self.assertEqual(report['excluded_images'],2)
        self.assertEqual(report['moved_images'],0)
        self.assertTrue(all(a['from_split']=='validation' for a in actions))
        self.assertEqual(result['test'],base['test'])
        self.assertEqual({i['id'] for i in result['validation']['images']},{20})
        manifest['reviewed_edges'].pop()
        with self.assertRaisesRegex(ValueError,'closure'):
            revise(base,manifest)


if __name__=='__main__':
    unittest.main()
