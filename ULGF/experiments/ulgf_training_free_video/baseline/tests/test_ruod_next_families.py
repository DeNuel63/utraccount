import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from tools.review_ruod_next_families import prepare
from tools.review_ruod_v3_candidates import compose_actions


class NextFamilyTests(unittest.TestCase):
    def test_compose_moved_then_excluded_in_original_namespace(self):
        result=compose_actions([[dict(member='train:2',to_split='validation')],
                                [dict(member='validation:2',to_split='exclude'),
                                 dict(member='validation:11',to_split='exclude')]])
        self.assertEqual({r['member']:r['to_split'] for r in result},
                         {'train:2':'exclude','train:11':'exclude'})

    def test_respects_v2_actions_paginates_and_does_not_approve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            screen=root/'screen'
            screen.mkdir()
            image=root/'image.jpg'
            Image.new('RGB',(20,20),'blue').save(image)
            right=dict(split='test',id=99,file_name='99.jpg',path=str(image))
            candidates=[]
            for i in range(1,11):
                left=dict(split='train',id=i,file_name=str(i)+'.jpg',path=str(image))
                candidates.append(dict(left=left,right=right,phash_distance=0,dhash_distance=0,score=0))
            candidates.append(dict(left=dict(split='validation',origin='train',id=11,
                file_name='11.jpg',path=str(image)),right=right,phash_distance=0,dhash_distance=0,score=0))
            (screen/'train__test.json').write_text(json.dumps(candidates))
            (screen/'validation__test.json').write_text('[]')
            actions=root/'actions.json'
            actions.write_text(json.dumps([dict(member='train:1',to_split='exclude'),
                                            dict(member='train:2',to_split='validation'),
                                            dict(member='train:11',to_split='exclude')]))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(screen,actions,root/'output')
            data=json.loads((root/'output/batch_candidates.json').read_text())
            self.assertEqual(len(data),1)
            self.assertNotIn('train:1',data[0]['shown_members'])
            self.assertNotIn('validation:11',data[0]['shown_members'])
            self.assertEqual(data[0]['shown_members']['train:2']['split'],'validation')
            self.assertEqual(data[0]['total_members'],10)
            self.assertEqual(data[0]['status'],'AWAITING_VISUAL_REVIEW')
            self.assertTrue((root/'output/B02F01_page2.png').is_file())
            self.assertEqual(len(json.loads((screen/'train__test.json').read_text())),11)


if __name__=='__main__':
    unittest.main()
