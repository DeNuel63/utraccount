import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from tools.staged_colab_recovery import (digest,make_shard,restore_shard,
    safe_target,publish_file,check_ready,atomic_json)


class StagedRecoveryTests(unittest.TestCase):
    def test_restore_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'a').write_bytes(b'image'); tar=root/'a.tar'
            record=make_shard(root,['a'],tar)
            restore_shard(tar,record,root/'out');restore_shard(tar,record,root/'out')
            self.assertEqual((root/'out/a').read_bytes(),b'image')

    def test_changed_existing_image_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').write_bytes(b'image');tar=root/'a.tar'
            record=make_shard(root,['a'],tar);(root/'out').mkdir();(root/'out/a').write_bytes(b'keep')
            with self.assertRaises(ValueError): restore_shard(tar,record,root/'out')
            self.assertEqual((root/'out/a').read_bytes(),b'keep')

    def test_corrupt_archive_rejected_before_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').write_bytes(b'image');tar=root/'a.tar'
            record=make_shard(root,['a'],tar);tar.write_bytes(b'bad')
            with self.assertRaises(ValueError):restore_shard(tar,record,root/'out')
            self.assertFalse((root/'out').exists())

    def test_traversal_and_links_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('../outside','/absolute','C:/outside','a\\b'):
                with self.assertRaises(ValueError):safe_target(root,name)
            tar=root/'link.tar'
            with tarfile.open(tar,'w') as z:
                info=tarfile.TarInfo('link');info.type=tarfile.SYMTYPE;info.linkname='../outside';z.addfile(info)
            with self.assertRaises(ValueError):restore_shard(tar,{'sha256':digest(tar),'files':{'link':{'size':0,'sha256':''}}},root/'out')

    def test_missing_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').write_bytes(b'image');tar=root/'a.tar'
            record=make_shard(root,['a'],tar);record['files']['missing']=record['files']['a']
            with self.assertRaises(ValueError):restore_shard(tar,record,root/'out')

    def test_publish_preserves_conflicting_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').write_bytes(b'new');(root/'b').write_bytes(b'old')
            with self.assertRaises(ValueError):publish_file(root/'a',root/'b')
            self.assertEqual((root/'b').read_bytes(),b'old')

    def test_ready_requires_matching_evidence_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):check_ready(root,'evidence')
            (root/'file').write_bytes(b'good')
            atomic_json(root/'READY.json',dict(status='CPU_PREPARATION_PASS',evidence_sha256='evidence',files={'file':digest(root/'file')}))
            self.assertEqual(check_ready(root,'evidence')['status'],'CPU_PREPARATION_PASS')
            with self.assertRaises(ValueError):check_ready(root,'different')
            (root/'file').write_bytes(b'bad')
            with self.assertRaises(ValueError):check_ready(root,'evidence')

    def test_shards_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a').write_bytes(b'image')
            x=make_shard(root,['a'],root/'one.tar');y=make_shard(root,['a'],root/'two.tar')
            self.assertEqual(x,y)


if __name__=='__main__':unittest.main()
