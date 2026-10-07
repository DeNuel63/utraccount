# ULGF training results

## Scope and conclusion

On 7 October 2026, run `20261007T103836275651Z` completed a **100-update
still-image engineering continuation** from existing RUOD-trained ULGF weights.
Training, inference export, backup and actual checkpoint resume were exercised.
This was not a full training campaign or a validated video-generation experiment.

**Meaningful quality improvement was not demonstrated.** The larger paired
evaluation found a small increase in denoising loss. Retain the trained checkpoint
as an engineering artifact, not as a demonstrated improvement over the original.

Results below come from supplied Colab execution reports and inspected images.
Detailed logs, review evidence and handover history remain in the private archive;
weights, dataset images and generated figures are not distributed in this document.

## Run configuration

| Setting | Value |
| --- | --- |
| Initialization | Existing RUOD-trained ULGF checkpoint; holdout provenance unresolved |
| Resolution | 256 × 256 |
| Optimizer / updates | AdamW8bit / 100 |
| Learning rate / schedule | `5e-6` / constant, no warmup |
| Batch size / gradient accumulation | 1 / 1 |
| Training seed | 42 |
| Precision / gradient checkpointing | fp16 / enabled |
| Trainable component | UNet; VAE and text encoder frozen |
| EMA | Disabled |
| Loss | Weighted epsilon MSE; constant foreground weight 1.0, normalization enabled |
| Dataset subset | 7,215 training images; 812 validation images |

The engineering subset retains complete annotations only when they fit both the
22-box and 77-token limits. It excludes crowded/long-prompt cases and cannot
establish full-dataset performance. It derives from reviewed v11 (7,515 training,
842 validation and 4,200 official-test images). The engineering run preserved the
original split and official test; test images were not used as validation.
Remaining leakage review and initialization provenance prevent a clean-holdout claim.

### Recorded environment

Tesla T4; Python 3.7.16; PyTorch 1.12.1+cu113; torchvision 0.13.1+cu113;
CUDA build 11.3; MMCV-full 1.7.0; MMDetection 2.25.3; Accelerate 0.20.3;
Diffusers 0.4.1; Transformers 4.24.0; bitsandbytes 0.35.4; NumPy 1.21.6;
Pillow 9.5.0. The training environment used OpenCV 4.10.0.84 and ftfy 6.1.1.

These describe the qualified Colab run, not an instruction to install every
package into a modern default notebook kernel. The upstream training requirements
are not a complete reproducibility lockfile. Preserve the run's package freeze,
source/annotation hashes and configuration alongside any reproduction.

## Evaluation

Lower weighted epsilon MSE is better. Both checkpoints used matched validation
inputs and shared stochastic settings; no optimizer updates occurred during comparison.

| Evaluation | Original loss | Step-100 loss | Relative change |
| --- | ---: | ---: | ---: |
| Four fixed examples, noise seed 12345 | 0.12232011 | 0.12222775 | 0.0755% lower |
| 64 images × three noise seeds | 0.16379158 | 0.16394605 | **0.0943% higher** |

The expanded test selected 64 images without replacement using selection seed
20261007, and used noise seeds 12345, 23456 and 34567: **192 paired probes**.
Hashes verified matching transformed pixels, tokens, masks, sampled latents,
noise and timesteps. The private sample manifest records image IDs, indices and
the validation-annotation SHA-256; per-image JSONL and paired CSV preserve losses.

Only 15/64 images improved after averaging seeds. All three seed-level mean losses
increased. The after-minus-before mean difference was 0.00015448; its image-bootstrap
95% interval was [0.00009478, 0.00021819]. This interval is conditional on the three
seeds and does not model dependence between related visual families.

The four-example trained loss reproduced exactly, but that smaller result did not
generalize to the expanded sample. Neither test is a perceptual-quality metric.

## Verification and limitations

- Complete inference export and backup manifest: server-side size/digest verification passed.
- Actual full-state resume from step 100 to 101: passed; parameters changed.
  Peak allocated GPU memory during that test: 10,491.25 MiB.
- Exact equivalence to uninterrupted training: **not tested**.
- Fixed-seed generated-image inspection: anatomy and object-clarity defects remained.
  Three unflagged pairs were numerically different; this did not demonstrate visual improvement.
- One post-training image was replaced with black by the enabled safety checker.
  It is not evidence of a black underlying model output.
- Requested bounding-box overlays do not independently establish localization accuracy.
- Video sequences, temporal consistency and persistent-ID correctness: **not validated**.

The follow-up's overall status was STOPPED after an artifact-transfer timeout.
Successful server receipts separately confirmed the complete inference backup,
resume evidence and comparison artifacts; the timeout is not hidden or relabelled.

## Obtain and load the checkpoint

Weights are not included in Git. Authorized collaborators should request access
from the project maintainer to the new-account Drive folder:

```text
My Drive/ULGF-trained-checkpoints/
  engineering-step100-20261007T103836275651Z/
```

Obtain the **entire directory**, including `backup-manifest.json`, model and
generation configuration, UNet, VAE, text encoder, tokenizer, scheduler, safety
checker and feature extractor. A UNet file alone is not a complete inference pipeline.
The original checkpoint is a different artifact; do not mix its components into
the trained export or rename it as the trained result.

Trained `unet/diffusion_pytorch_model.bin` SHA-256:

```text
640241e953e5203388ae3471b1df91258e413af04df056a309cd9c2e8297241a
```

Before loading, verify every exported file against the SHA-256 values in the
trusted backup manifest's `files` mapping. A manifest supplied by an untrusted
party is not independent provenance. Load pickle-based model files only from
trusted sources. Access to the backup does not establish redistribution rights;
retain upstream notices and check checkpoint/dataset terms before sharing.

Use the existing pinned inference environment described in the [README](../README.md#environment).
Install its inference dependencies from [requirements-inference.txt](../requirements-inference.txt)
after the compatible PyTorch build. Do not install overlapping OpenCV distributions
into an already qualified training environment.

Keep the complete checkpoint at a Git-ignored location such as
`assets/checkpoint/engineering-step100/`, or supply an external absolute path.
From the repository root, validate paired input images and YOLO labels first:

```sh
python scripts/generate_image_baseline.py \
  --checkpoint /absolute/path/to/engineering-step100-20261007T103836275651Z \
  --image-dir /absolute/path/to/images \
  --label-dir /absolute/path/to/labels \
  --output-dir work_dirs/step100-inference \
  --device cuda --seed 1000 --guidance-scale 5.0 --inference-steps 100 \
  --dry-run
```

Replace the input paths. Labels must use the RUOD class order expected by the
entry point, or pass an explicitly matching `--classes` mapping. Remove only
`--dry-run` to load the pipeline and generate. Keep the safety checker enabled;
use a fresh output directory rather than overwriting previous results. A dry run
validates the plan but does not prove successful model deserialization or image quality.

The inference backup does **not** contain a complete optimizer/resume snapshot.
The tested training state was runtime-local and may be lost after Colab reset.
Starting from inference weights alone resets optimizer state and is a new
continuation experiment, not exact resume.

## Private evidence locations

Under `My Drive/ULGF-trained-checkpoints/reports/engineering-training-20261007T103836275651Z/`:

- `followup-20261007T105307369610Z/`: backup/resume evidence and visual comparison.
- `validation-comparison-20261007T113310386143Z/`: four-example comparison.
- `validation64-20261007T114436666308Z/`: 64-image manifest, paired losses and summary.

These locations identify supporting evidence; they are not public download links.
The detailed handover and progress checklist remain under `_local_archive/`,
excluded from the source upload. No claim of complete public reproducibility is
made without the corresponding authorized inputs, manifests and recorded source versions.
