"""CPU-only playback exports from existing verified frames; no interpolation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from PIL import Image,ImageSequence

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/reference_plate_cpu_20260929T223947096913Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    report=json.loads((RUN/'qc.json').read_text())
    paths=[RUN/'frames'/('%06d.png'%i) for i in range(5)]
    hashes=[sha(p) for p in paths]
    assert hashes==[r['frame_sha256'] for r in report['frames']]
    frames=[Image.open(p).convert('RGB') for p in paths]
    assert all(f.size==(256,256) for f in frames)
    out=ROOT/'previews'/('clean_plate_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    durations=[250,250,250,250,750]
    # Lossless WebP is the fidelity reference. GIF is a convenient reduced-palette copy.
    frames[0].save(out/'playback_lossless.webp',save_all=True,append_images=frames[1:],
                   duration=durations,loop=0,lossless=True,quality=100,method=6)
    atlas=Image.new('RGB',(256*5,256))
    for i,f in enumerate(frames):atlas.paste(f,(i*256,0))
    palette=atlas.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    indexed=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
    indexed[0].save(out/'playback.gif',save_all=True,append_images=indexed[1:],
                    duration=durations,loop=0,optimize=False,disposal=2)
    with Image.open(out/'playback_lossless.webp') as animation:
        assert animation.n_frames==5
        for i in range(5):
            animation.seek(i)
            assert np.array_equal(np.asarray(animation.convert('RGB')),np.asarray(frames[i]))
    with Image.open(out/'playback.gif') as animation:
        assert animation.n_frames==5
        actual=[]
        for i in range(5):animation.seek(i);actual.append(animation.info['duration'])
        assert actual==durations
    assert hashes==[sha(p) for p in paths]
    metadata=dict(source_run=RUN.name,source_frame_sha256=hashes,
        user_composition_approval='User: It is acceptable',frames=5,frame_duration_ms=durations,
        loop_duration_ms=sum(durations),interpolation=False,generated_frames=False,
        playback='Forward 0-4, hold final frame, then jump back to frame 0; reset is not continuous swimming',
        lossless_webp_pixel_verification='PASS',gif='Shared 256-colour palette; not pixel-exact',
        source_unchanged=True,training_started=False,gpu_used=False)
    (out/'preview_info.json').write_text(json.dumps(metadata,indent=2))
    print(out.name);print('PASS: 5 WebP frames pixel-identical; GIF timings verified; originals unchanged.')

if __name__=='__main__':main()
