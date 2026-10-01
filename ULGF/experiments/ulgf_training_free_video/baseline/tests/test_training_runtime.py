import json
from pathlib import Path
import tempfile
import unittest
from ulgf_baseline.reviewed_data import reviewed_paths
from ulgf_baseline.training_runtime import EpochSampler,check_contract,resolve_checkpoint

class RuntimeContractTests(unittest.TestCase):
    def test_numpy_rng_primitive_roundtrip_preserves_cached_gaussian(self):
        import random
        import numpy as np
        from ulgf_baseline.training_runtime import encode_rng,decode_rng
        rng=np.random.RandomState(42)
        rng.normal()  # Cache the second Gaussian draw.
        state=dict(python=random.getstate(),numpy=rng.get_state(),cpu=None,cuda=[])
        encoded=encode_rng(state)
        self.assertIsInstance(encoded['numpy']['keys'],list)
        restored=decode_rng(encoded)
        other=np.random.RandomState(); other.set_state(restored['numpy'])
        self.assertEqual(other.normal(),rng.normal())
        self.assertEqual(other.randint(100000),rng.randint(100000))
        self.assertEqual(restored['python'],state['python'])

    def test_sampler_mid_epoch_remainder_and_epoch_change(self):
        sampler=EpochSampler(19,42)
        order=list(sampler)
        self.assertEqual(set(order),set(range(19)))
        sampler.offset=8
        self.assertEqual(list(sampler),order[8:])
        self.assertEqual(len(sampler),11)
        sampler.offset=0; sampler.epoch=1
        self.assertNotEqual(list(sampler),order)

    def test_changed_resume_contract_rejected(self):
        check_contract({'split':'v6'},{'split':'v6'})
        with self.assertRaisesRegex(ValueError,'split'):
            check_contract({'split':'v5'},{'split':'v6'})

    def test_latest_ignores_incomplete_state(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            for n in (2,10,11):
                (root/('step-'+str(n))).mkdir()
            for n in (2,10):
                (root/('step-'+str(n))/'complete.json').write_text('{}')
            self.assertEqual(resolve_checkpoint(root,'latest').name,'step-10')
            with self.assertRaises(ValueError):
                resolve_checkpoint(root,str(root/'step-11'))

    def fixture(self,root):
        images=root/'images'; images.mkdir()
        (images/'1.jpg').touch(); (images/'2.jpg').touch()
        (root/'annotations').mkdir()
        (root/'paths.json').write_text(json.dumps({'image_prefix':str(images)}))
        for split,i in (('train',1),('validation',2)):
            (root/'annotations'/('instances_'+split+'.json')).write_text(json.dumps({'images':[{'id':i,'file_name':str(i)+'.jpg'}]}))
        return root

    def test_reviewed_paths_and_test_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.fixture(Path(tmp))
            result=reviewed_paths(root)
            self.assertTrue(result['validation'].endswith('instances_validation.json'))
            (root/'annotations/instances_test.json').write_text((root/'annotations/instances_validation.json').read_text())
            with self.assertRaisesRegex(ValueError,'Official test'):
                reviewed_paths(root)

    def test_missing_image_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=self.fixture(Path(tmp))
            (root/'images/2.jpg').unlink()
            with self.assertRaises(FileNotFoundError):
                reviewed_paths(root)

try:
    import torch
    from accelerate import Accelerator
    TENSORS=True
except ImportError:
    TENSORS=False

@unittest.skipUnless(TENSORS,'Requires torch + accelerate; run in restored Colab environment')
class TensorRuntimeTests(unittest.TestCase):
    def test_rng_snapshot_loads_with_restricted_unpickler(self):
        from ulgf_baseline.training_runtime import capture_rng,encode_rng,load_rng_file
        state=capture_rng()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'rng.pt'
            torch.save(encode_rng(state),path)
            restored=load_rng_file(path)
            self.assertTrue(torch.equal(restored['cpu'],state['cpu']))
            self.assertEqual(restored['numpy'][1].tolist(),state['numpy'][1].tolist())

    def test_save_resume_matches_next_adam_update(self):
        from ulgf_baseline.training_runtime import save_state,load_state,capture_rng
        class Tokenizer:
            def save_pretrained(self,path):
                Path(path).mkdir(); (Path(path)/'tokenizer.json').write_text('{}')
        torch.manual_seed(7)
        accelerator=Accelerator(cpu=True)
        model=torch.nn.Linear(3,1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        model,optimizer=accelerator.prepare(model,optimizer)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
        accelerator.register_for_checkpointing(scheduler)
        def step():
            x=torch.randn(4,3)
            loss=model(x).square().mean()
            accelerator.backward(loss); optimizer.step(); scheduler.step(); optimizer.zero_grad()
            return loss.item()
        step()
        with tempfile.TemporaryDirectory() as root:
            path=save_state(accelerator,root,dict(global_step=1,epoch=0,next_batch=1),{'split':'fixture'},Tokenizer())
            expected_loss=step()
            expected=[p.detach().clone() for p in model.parameters()]
            expected_lr=scheduler.get_last_lr()
            progress=load_state(accelerator,path,{'split':'fixture'})
            self.assertEqual(progress['next_batch'],1)
            self.assertEqual(step(),expected_loss)
            self.assertEqual(scheduler.get_last_lr(),expected_lr)
            for p,e in zip(model.parameters(),expected):
                self.assertTrue(torch.equal(p,e))

    def test_validation_restores_rng_and_modes(self):
        from types import SimpleNamespace
        from ulgf_baseline.training_runtime import validate
        class VAE:
            def encode(self,x):
                return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda:x))
        class Text(torch.nn.Module):
            def forward(self,x): return (x.float(),)
        class UNet(torch.nn.Module):
            def forward(self,x,t,h): return SimpleNamespace(sample=x*0)
        class Scheduler:
            num_train_timesteps=10
            def add_noise(self,x,n,t): return x+n
        accelerator=Accelerator(cpu=True)
        unet,text=UNet(),Text()
        batch=dict(pixel_values=torch.ones(2,3,4,4),input_ids=torch.ones(2,3,dtype=torch.long),bbox_mask=None)
        before=torch.get_rng_state().clone()
        a=validate(unet,text,VAE(),Scheduler(),[batch],accelerator,torch.float32,123)
        b=validate(unet,text,VAE(),Scheduler(),[batch],accelerator,torch.float32,123)
        self.assertEqual(a,b)
        self.assertTrue(torch.equal(torch.get_rng_state(),before))
        self.assertTrue(unet.training and text.training)
