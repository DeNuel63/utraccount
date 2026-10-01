"""Single-process deterministic validation and complete training-state snapshots.

Tensor imports are lazy so contract tests also run without PyTorch.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
from .reviewed_data import sha256


class EpochSampler:
    def __init__(self, size, seed):
        self.size,self.seed,self.epoch,self.offset=size,seed,0,0
    def __iter__(self):
        order=list(range(self.size))
        random.Random(self.seed+self.epoch).shuffle(order)
        return iter(order[self.offset:])
    def __len__(self):
        return self.size-self.offset


class SeededDataset:
    """Per-epoch/item deterministic transforms; prefetch cannot advance training RNG."""
    def __init__(self, dataset, seed):
        self.dataset,self.seed,self.epoch=dataset,seed,0
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,index):
        import numpy as np
        import torch
        py,np_state,cpu=random.getstate(),np.random.get_state(),torch.get_rng_state()
        try:
            seed=(self.seed+self.epoch*1000003+index)%(2**32)
            random.seed(seed); np.random.seed(seed)
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            return self.dataset[index]
        finally:
            random.setstate(py); np.random.set_state(np_state); torch.set_rng_state(cpu)


def capture_rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda'])!=torch.cuda.device_count():
            raise ValueError('CUDA RNG device topology changed')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def encode_rng(state):
    """Use only tensors and primitive containers, not pickled NumPy objects."""
    algorithm,keys,position,has_gauss,cached_gaussian=state['numpy']
    return dict(format_version=1,python=state['python'],cpu=state['cpu'],cuda=state['cuda'],
                numpy=dict(algorithm=algorithm,keys=keys.tolist(),position=int(position),
                           has_gauss=int(has_gauss),cached_gaussian=float(cached_gaussian)))


def decode_rng(payload):
    import numpy as np
    if payload.get('format_version')!=1:
        raise ValueError('Unsupported RNG snapshot format; no unsafe pickle fallback')
    state=payload['numpy']
    return dict(python=payload['python'],cpu=payload['cpu'],cuda=payload['cuda'],
        numpy=(state['algorithm'],np.asarray(state['keys'],dtype=np.uint32),
               state['position'],state['has_gauss'],state['cached_gaussian']))


def load_rng_file(path):
    import inspect
    import torch
    options=dict(map_location='cpu')
    # Older pinned torch releases may lack the keyword; new releases stay restricted.
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only']=True
    return decode_rng(torch.load(path,**options))


@contextmanager
def fixed_rng(seed):
    import numpy as np
    import torch
    state=capture_rng()
    try:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        yield
    finally:
        restore_rng(state)


def validate(unet,text_encoder,vae,scheduler,loader,accelerator,dtype,seed,max_batches=0):
    import torch
    modes=(unet.training,text_encoder.training)
    total,count=0.,0
    try:
        unet.eval(); text_encoder.eval()
        with fixed_rng(seed),torch.no_grad():
            for index,batch in enumerate(loader):
                if max_batches and index>=max_batches:
                    break
                with accelerator.autocast():
                    pixels=batch['pixel_values'].to(accelerator.device,dtype=dtype)
                    latents=vae.encode(pixels).latent_dist.sample()*0.18215
                    noise=torch.randn_like(latents)
                    timesteps=torch.randint(0,scheduler.num_train_timesteps,(len(latents),),device=latents.device).long()
                    hidden=text_encoder(batch['input_ids'].to(accelerator.device))[0]
                    prediction=unet(scheduler.add_noise(latents,noise,timesteps),timesteps,hidden).sample
                    error=(prediction.float()-noise.float()).square()
                    mask=batch['bbox_mask']
                    if mask is not None:
                        mask=mask.to(accelerator.device)
                        if mask.shape!=(len(latents),1,error.shape[2],error.shape[3]) or not torch.isfinite(mask).all() or (mask<=0).any():
                            raise ValueError('Invalid validation mask')
                        error=error*mask
                    losses=error.flatten(1).mean(1)
                    if not torch.isfinite(losses).all():
                        raise FloatingPointError('Nonfinite validation loss')
                    total+=losses.sum().item(); count+=len(losses)
        if not count:
            raise ValueError('Empty validation loader')
        return dict(loss=total/count,examples=count,seed=seed,metric='held_out_weighted_epsilon_mse',weights='online')
    finally:
        unet.train(modes[0]); text_encoder.train(modes[1])


def check_contract(saved,expected):
    if saved!=expected:
        changed=sorted(k for k in set(saved)|set(expected) if saved.get(k)!=expected.get(k))
        raise ValueError('Resume contract differs: '+', '.join(changed))


def resolve_checkpoint(root, requested):
    if requested!='latest':
        path=Path(requested)
        if not (path/'complete.json').is_file():
            raise ValueError('Incomplete training state; inference exports cannot resume')
        return path
    candidates=[p for p in Path(root).glob('step-*') if (p/'complete.json').is_file()]
    if not candidates:
        raise ValueError('No complete training state')
    return max(candidates,key=lambda p:int(p.name.split('-')[-1]))


def save_state(accelerator,root,progress,contract,tokenizer):
    import torch
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    destination=root/('step-{:08d}'.format(progress['global_step']))
    if destination.exists():
        raise FileExistsError('Refusing checkpoint overwrite: '+str(destination))
    import tempfile
    temporary=Path(tempfile.mkdtemp(prefix='.incomplete-',dir=str(root)))
    rng=capture_rng()
    try:
        accelerator.save_state(str(temporary))
        torch.save(encode_rng(rng),temporary/'strict_rng.pt')
        tokenizer.save_pretrained(str(temporary/'tokenizer'))
        payload=dict(progress=progress,contract=contract,accelerator_step=accelerator.step)
        (temporary/'training.json').write_text(json.dumps(payload,indent=2))
        files={str(p.relative_to(temporary)):sha256(p) for p in temporary.rglob('*') if p.is_file()}
        (temporary/'complete.json').write_text(json.dumps(files,indent=2))
        temporary.rename(destination)
    finally:
        restore_rng(rng)
    return destination


def load_state(accelerator,path,contract):
    import torch
    path=Path(path)
    files=json.loads((path/'complete.json').read_text())
    for name,expected in files.items():
        target=(path/name).resolve(); target.relative_to(path.resolve())
        if sha256(target)!=expected:
            raise ValueError('Checkpoint integrity failure: '+name)
    payload=json.loads((path/'training.json').read_text())
    check_contract(payload['contract'],contract)
    accelerator.load_state(str(path))
    # Accelerate 0.20 can silently swallow RNG-load failures; enforce our own copy.
    restore_rng(load_rng_file(path/'strict_rng.pt'))
    accelerator.step=payload['accelerator_step']
    return payload['progress']
