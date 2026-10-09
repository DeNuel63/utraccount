# Model entry points and separate runtime requirements

Inspection date: 7 October 2026. These are identified sources and requirements,
not a claim that all three GPU environments have been installed or qualified.

## Inference entry points and checkpoints

| Stage | Inference function used by the connection | Checkpoint |
| --- | --- | --- |
| ULGF | `ULGFAdapter.generate()` in `ULGF/experiments/ulgf_training_free_video/run_five_frames.py`; loads through `utils.generation_utils.load_checkpoint()` | Complete Diffusers directory with generation config, UNet, VAE, text encoder, tokenizer, scheduler, safety checker and feature extractor. Local candidate: `ULGF/assets/checkpoint/final/`. All 17 required component files exist and are non-empty; loading is unverified. |
| CountGD++ | `CountGDPlusPlusBackend.infer()` in `COUNTGD++/src/utraccount_count/countgdplusplus.py`; loads `app.build_model_and_transforms()` and invokes its model | Official `countgd_plusplus.pth`, whose top-level `model` key is consumed by `app.py`. Requires BERT assets at `<official repository>/checkpoints/bert-base-uncased/` as well. Neither artifact was found locally. User reports weights on Drive; exact path remains unresolved. |
| COVTrack | `init_model()` and `inference_model(..., external_detections=...)` in `CovTrack_Model/COVTrack/ovtrack/apis/inference.py`; dispatches to `track_external_detections()` | Published `saved_models/ctao_public_res/ctao_public.pth`, paired with `configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py`. Published artifact identified; not downloaded or loaded locally. |

The local ULGF UNet SHA-256 is
`6f1ee83477953428b3cfc95769b7cfa1fdc5321b598b211192ecb25424f53bff`.
This differs from the step-100 export hash recorded in `ULGF/docs/TRAINING_RESULTS.md`.
The local directory must not be described as that trained export. The user's
intended Drive checkpoint still needs to be identified before choosing weights.

Published sources:

- CountGD++: https://drive.google.com/file/d/1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs/view
  (linked in the supplied official README).
- COVTrack: https://huggingface.co/clarkqian/COVTrack/blob/main/saved_models/ctao_public_res/ctao_public.pth
  (369 MB shown in the model repository).

## Use three separate environments

| Stage | Documented environment requirements | Evidence level |
| --- | --- | --- |
| ULGF inference | Python 3.7.16; torch 1.12.1+cu113; torchvision 0.13.1; diffusers 0.4.1; transformers 4.24.0; accelerate 0.20.3; pinned `ULGF/requirements-inference.txt` | Repository records a previously exercised Colab environment. Not available in this desktop runtime. |
| CountGD++ | Python 3.10; torch <2.6; matching torchvision; Gradio >=4,<5; Detectron2; compiled MultiScaleDeformableAttention; GCC 11.3/11.4 and CUDA toolkit per upstream README | Upstream instructions; torch/torchvision/transformers are not fully locked. Exact resolved versions and compiled operators must be checked in the target GPU runtime. |
| COVTrack | torch >=1.10; MMCV 1.4.4 with compiled operators; MMDetection 2.23; CLIP plus repository requirements | Supplied installation guide. Python/CUDA/exact torch versions are not pinned by that guide. Select a matching compiled MMCV wheel and qualify the model in its own GPU environment. |

ULGF's recorded Python 3.7 stack and CountGD++'s Python 3.10 stack cannot be one
Python environment. Keep COVTrack's old MMDetection/MMCV operators isolated as
well. Do not substitute current MMDetection 3.x or MMCV 2.x for the documented
APIs. The launcher already accepts a separate executable for every stage and
uses sequential processes and JSON/image handoffs.

The bundled desktop Python is for inspection and the launcher, not model inference.
It lacks torch, torchvision, MMCV, MMDetection, Detectron2, Diffusers and Transformers.
No model runtime was installed by this inspection.

## Check the actual environments

Run the following with each stage's actual interpreter after mounting Drive or
restoring weights. Capture each JSON result alongside the run configuration:

```sh
/path/to/ulgf/python integration/runtime_probe.py --stage ulgf
/path/to/countgd/python integration/runtime_probe.py --stage countgd
/path/to/covtrack/python integration/runtime_probe.py --stage covtrack
```

`IMPORT_PROBE_PASS` means required imports and CUDA detection succeeded. It does
not confirm operator numerical correctness, checkpoint compatibility, or inference.
For CountGD++, also run the upstream `models/GroundingDINO/ops/test.py` in its
environment. Final compatibility confirmation requires the existing pipeline
runner to load all intended checkpoints and process a short clip successfully.
The supplied `config.example.json` contains illustrative paths; it is not the
user's resolved Drive configuration.

## Shared Drive inspection on 8 October 2026

`drive_assets.json` records the folder names and checkpoint entries observed through
the supplied shared links. `CountGD/countgd_plusplus.pth` is visible (1.16 GB).
`ULGF-assets/checkpoints/` contains two engineering step-100 directories dated
5 and 6 October plus a `ruod_256_bz16` shortcut. Their contents were not loaded,
and no checkpoint was silently substituted for the intended ULGF model.

Folder display names do not establish mounted `/content/drive/MyDrive` paths.
The shared browser session was signed out; both supplied Colab pages failed to
load into a usable notebook, and Google sign-in failed with a name-resolution
error. No GPU runtime was connected and no models executed during this inspection.
Execution requires an accessible signed-in GPU notebook or another authenticated
GPU executor with the repository and checkpoint assets available.
