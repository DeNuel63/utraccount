"""Guarded experimental source transform; never writes the frozen baseline."""
import ast

TARGET = '        timesteps_tensor = timesteps_tensor[5:]'
REPLACEMENT = ('        if latents_prior_flag:\n'
               '            timesteps_tensor = timesteps_tensor[5:]')

def corrected_source(source):
    if source.splitlines().count(TARGET) != 1:
        raise ValueError('Expected exactly one frozen timestep-slicing statement')
    changed = source.replace(TARGET, REPLACEMENT, 1)
    ast.parse(changed)
    return changed

def corrected_class(module):
    from pathlib import Path
    source = Path(module.__file__).read_text(encoding='utf-8')
    namespace = dict(vars(module))
    exec(compile(corrected_source(source), '<experimental-no-prior-schedule>', 'exec'), namespace)
    return namespace['PriorDiffusionPipeline']
