# ULGF Video Extension Engineering Roadmap

## 1. Objective

Extend the existing still-image ULGF repository into a synthetic underwater clip generator while preserving the original image-generation baseline. The completed system must generate an ordered sequence of frames together with per-frame bounding boxes, class labels, and persistent object instance IDs.

This roadmap covers ULGF only. CountGD++ and COVTrack are downstream consumers and are not implementation targets here.

## 2. Definition of Done

The video extension is complete when a single command can accept a seed image and its annotations, generate a deterministic synthetic clip, and export:

- `T` ordered PNG frames;
- one clip-level manifest;
- normalized `xyxy` boxes and class labels for every visible object in every frame;
- a persistent instance ID that remains stable across the clip;
- generation metadata including seed, model checkpoint, configuration, frame size, and frame rate;
- an optional MP4 preview derived from the PNG frames;
- validation and evaluation reports showing that the clip and annotations satisfy the contract below.

The image sequence and manifest are the canonical outputs. A compressed video is a convenience artifact, not the source of truth.

## 3. Canonical Output Contract

Use a versioned JSON manifest. Coordinates are normalized to `[0, 1]` and use `xyxy` order.

```json
{
  "schema_version": "1.0",
  "clip_id": "clip_000001",
  "seed": 1234,
  "fps": 10,
  "width": 256,
  "height": 256,
  "medium": {
    "description": "blue-green turbid water with diffuse lighting"
  },
  "frames": [
    {
      "frame_index": 0,
      "image_path": "frames/000000.png",
      "objects": [
        {
          "instance_id": 1,
          "class_id": 4,
          "class_name": "fish",
          "bbox_xyxy": [0.12, 0.31, 0.27, 0.46],
          "visibility": 1.0
        }
      ]
    }
  ],
  "generation": {
    "checkpoint": "path-or-model-id",
    "config": "configs/video/ruod_video_256x256.py",
    "temporal_mode": "independent"
  }
}
```

Contract invariants:

- `frame_index` values are contiguous and start at zero.
- Every referenced frame exists and has the declared dimensions.
- Every box satisfies `0 <= x1 < x2 <= 1` and `0 <= y1 < y2 <= 1`.
- An `instance_id` maps to one class for its entire lifetime.
- An object may be absent because of entry, exit, or occlusion, but an ID is never reassigned.
- Re-running with the same checkpoint, configuration, and seed produces the same layouts and, within supported deterministic CUDA limits, the same frames.

## 4. Target Repository Structure

Add new video functionality without turning the existing entry points into larger monoliths.

```text
configs/
  video/
    ruod_video_256x256.py
pipeline_custom/
  pipeline_ULGF_video.py
ulgf_video/
  __init__.py
  schema.py
  layout_sequence.py
  motion.py
  conditioning.py
  latent_warp.py
  generator.py
  export.py
  validation.py
  metrics.py
scripts/
  generate_video_clip.py
  visualize_layout_sequence.py
  validate_video_clip.py
  evaluate_video_clips.py
tests/
  test_schema.py
  test_motion.py
  test_layout_sequence.py
  test_conditioning.py
  test_latent_warp.py
  test_export.py
  test_validation.py
  integration/
    test_clip_generation_smoke.py
```

The new package name is `ulgf_video`.

Existing files to modify conservatively:

| File | Planned change |
|---|---|
| `utils/generation_utils.py` | Expose reusable prompt, layout, checkpoint, and deterministic-seed helpers without changing still-image behavior. |
| `pipeline_custom/pipeline_prior.py` | Factor reusable latent preparation and denoising hooks if needed; preserve the public still-image call. |
| `train_UWLGM.py` | Add temporal-training hooks only after inference baselines pass; the current diffusion loss is near lines 732-738, not line 34. |
| `tools/dist_train.sh` | Remove machine-specific assumptions and make GPU/process count configurable. |
| `requirements.txt` | Add missing direct dependencies and document pinned baseline versus video-extension dependencies. |
| `README.md` | Add safe baseline and video-generation instructions after commands are verified. |

## 5. Phase Plan

### Phase 0 Baseline Reproduction and Safety

**Purpose:** establish a trustworthy still-image baseline before adding video behavior.

Split this phase operationally into Phase 0A, which hardens the repository and can be completed without model assets, and Phase 0B, which reproduces the baseline on the pinned CUDA environment. Phase 0 is not complete until both parts pass.

#### Modules and files

- Refactor `run_dataset_expansion.py` into a safe, argument-driven entry point or add `scripts/generate_image_baseline.py` around its reusable functions.
- Update `tools/dist_train.sh` to accept `NUM_PROCESSES`, `GPU_IDS`, model path, dataset config, and output directory.
- Update `requirements.txt` to include direct imports such as `accelerate` and any visualization packages actually required.
- Add `configs/baseline/ruod_local.example.py` or document environment-variable overrides rather than committing machine paths.
- Add `tests/integration/test_image_generation_smoke.py` using a tiny fixture or mocked diffusion components.

#### Work items

1. Inventory and document all external checkpoints and datasets.
2. Remove default paths under `/mnt/data0` and `/home/mly` from runnable commands.
3. Make deletion opt-in. No generation command may delete existing images or labels by default.
   Generated artifacts must use a dedicated output root that is not nested inside the source images, labels, or checkpoint.
4. Reconcile the claimed single-GPU setup with `tools/dist_train.sh`, which currently launches eight processes on GPU 0.
5. Record the exact baseline environment and the output of `pip freeze` or `conda env export`.
6. Generate a fixed baseline sample set and record seeds, prompts, layouts, and checksums.
7. Keep the Stable Diffusion safety checker enabled by default. Any opt-out must be explicit and recorded in run metadata.

#### Acceptance tests

- `--help` works without importing a dataset or loading a model.
- A dry run validates paths and configuration without writing output.
- A run cannot overwrite or delete an existing output unless an explicit flag is provided.
- The same seed produces the same encoded layout and initial latent tensor.
- The baseline generation command produces at least one valid image and matching annotation on the target GPU environment.
- All existing Python files still compile after refactoring.

#### Deliverables

- Reproducible environment specification.
- Safe still-image generation command.
- Baseline sample pack with configuration and checksums.
- Baseline run report containing runtime, peak VRAM, image size, and checkpoint identity.

#### Exit gate

Phase 0A may finish before external assets are available, but do not begin Phase 1 until Phase 0B has generated a stock-style image twice from a clean environment with matching inputs and recorded outputs.

### Phase 1 Clip Schema and Persistent Identity

**Purpose:** make video, frame, and identity concepts first-class before implementing motion or temporal diffusion.

#### Modules and files

- `ulgf_video/schema.py`
- `ulgf_video/export.py`
- `ulgf_video/validation.py`
- `tests/test_schema.py`
- `tests/test_export.py`
- `tests/test_validation.py`
- `docs/video_output_schema.md`

#### Work items

1. Define typed structures for `ClipSpec`, `FrameLayout`, `ObjectInstance`, `MediumSpec`, and generation metadata.
2. Assign persistent IDs deterministically when importing an initial layout.
3. Implement JSON serialization, schema versioning, and strict validation.
4. Implement optional YOLO export as a derived format. Because ordinary YOLO labels have no instance-ID field, write a documented sidecar mapping or use a tracking-compatible format.
5. Add forward-compatibility rules for new optional fields.

#### Acceptance tests

- Valid manifests round-trip through serialization without information loss.
- Invalid coordinates, duplicate IDs within a frame, class changes for an ID, missing frames, and path traversal are rejected.
- Unknown optional fields do not break readers of the same major schema version.
- A seed layout receives stable, deterministic instance IDs.
- Manifest paths are relative to the clip directory and portable across machines.

#### Deliverables

- Versioned schema and examples.
- Schema validator CLI.
- Unit tests for valid and invalid manifests.
- Conversion notes for COCO, YOLO, and MOT-style consumers.

#### Exit gate

No later phase may introduce its own ad hoc layout dictionary. All video code must use the Phase 1 structures.

### Phase 2 Sequential Layout Generator

**Purpose:** generate plausible, deterministic object trajectories without invoking diffusion.

#### Modules and files

- `ulgf_video/motion.py`
- `ulgf_video/layout_sequence.py`
- `scripts/visualize_layout_sequence.py`
- `configs/video/ruod_video_256x256.py`
- `tests/test_motion.py`
- `tests/test_layout_sequence.py`

#### Work items

1. Implement a constant-velocity model with bounded random acceleration.
2. Scale maximum displacement by box size and frame rate so small objects do not move implausibly fast.
3. Support configurable boundary policies: reflect, clamp, exit, and optional re-entry.
4. Preserve box width and height initially; add controlled scale change only after the translation model passes.
5. Implement collision awareness as a configurable soft repulsion rather than a hard guarantee.
6. Render layouts over a blank canvas and optionally over the seed image.
7. Record motion parameters in the manifest.

#### Acceptance tests

- The same seed and configuration produce byte-identical manifests.
- Every generated box remains valid under the configured boundary policy.
- Persistent IDs and classes remain unchanged.
- Per-frame displacement never exceeds the configured speed limit.
- Velocity changes stay within the configured acceleration limit.
- A zero-motion configuration produces identical layouts for all frames.
- Visual review of a fixed test set finds no teleportation or systematic boundary sticking.

#### Deliverables

- Sequential layout generator API and CLI.
- Layout-only preview videos for at least ten representative seed scenes.
- Motion configuration with documented units and defaults.
- Motion QA report listing observed failure cases.

#### Exit gate

Approve motion visually and through invariants before connecting it to image generation.

### Phase 3 Independent Frame Video Baseline

**Purpose:** create an end-to-end clip generator using the existing image model independently for each frame. This is the control condition for later temporal improvements.

#### Modules and files

- `ulgf_video/generator.py`
- `scripts/generate_video_clip.py`
- `scripts/validate_video_clip.py`
- `tests/integration/test_clip_generation_smoke.py`
- Minimal reusable changes in `utils/generation_utils.py`

#### Work items

1. Convert each `FrameLayout` into the existing ULGF prompt and layout representation.
2. Generate frames in index order using deterministic per-frame seeds derived from the clip seed.
3. Save PNG frames atomically and write the manifest only after all required frames succeed.
4. Support resume into an incomplete clip without regenerating completed, validated frames.
5. Produce an optional MP4 preview after the canonical frame sequence is complete.
6. Record timing and peak VRAM.

#### Acceptance tests

- A short clip of at least four frames is generated from one command.
- Every manifest frame has a corresponding readable image with correct dimensions.
- Interrupted generation can resume without changing completed frames.
- Partial output is clearly marked incomplete and is rejected by the normal validator.
- Still-image generation remains unchanged for the same prompt, layout, seed, and pipeline.
- The independent-frame mode is explicitly recorded as `temporal_mode: independent`.

#### Deliverables

- End-to-end independent-frame clip generator.
- Validated sample clips and preview videos.
- Baseline temporal-quality measurements.
- Runtime and VRAM benchmark.

#### Exit gate

This phase is the minimum viable video pipeline. Preserve its outputs as the comparison baseline for every later phase.

### Phase 4 Shared Clip Medium Conditioning

**Purpose:** reduce water-color, turbidity, and lighting drift while retaining frame-specific layouts.

#### Modules and files

- `ulgf_video/conditioning.py`
- `pipeline_custom/pipeline_ULGF_video.py`
- `tests/test_conditioning.py`
- Configuration additions under `configs/video/`

#### Work items

1. Split conditioning into clip-level medium text and frame-level object/layout text.
2. Cache the medium tokenization and embedding once per clip.
3. Define and test a combination strategy compatible with the pinned CLIP encoder: token concatenation, embedding concatenation with projection, or controlled prompt templating.
4. Detect and report token truncation rather than silently dropping layout tokens.
5. Add an ablation flag that disables shared conditioning.

#### Acceptance tests

- The text encoder computes the medium component once per clip, verified with instrumentation.
- Frame-specific layout tokens differ when layouts differ.
- Medium conditioning remains identical across all frames in a clip.
- Prompts exceeding the token budget fail validation or apply a documented deterministic compaction rule.
- A fixed evaluation set shows improved clip-level color and illumination consistency without materially reducing layout adherence.

#### Deliverables

- Shared-conditioning implementation.
- Prompt and embedding design note.
- Enabled-versus-disabled ablation report.
- Updated sample clips.

#### Exit gate

Proceed only if shared conditioning measurably reduces appearance drift or is shown not to harm the independent baseline.

### Phase 5 Latent Propagation and Warping

**Purpose:** introduce temporal dependence with limited architectural change and no required full-model retraining.

#### Modules and files

- `ulgf_video/latent_warp.py`
- `pipeline_custom/pipeline_ULGF_video.py`
- `tests/test_latent_warp.py`
- `tests/integration/test_clip_generation_smoke.py`

#### Work items

1. Convert object box displacement into a latent-resolution motion field.
2. Warp the previous frame's latent within object masks.
3. Define background handling and overlap blending.
4. Mix warped latent content with fresh seeded noise using a configurable strength.
5. Support keyframes or periodic latent reset to limit accumulated artifacts.
6. Expose independent, shared-noise, and warped-latent modes for ablation.

#### Acceptance tests

- Zero displacement produces an identity warp within numerical tolerance.
- Integer test displacements move synthetic latent patterns to expected positions.
- Warped latents have the expected shape, device, and dtype.
- Overlap blending never produces NaN or infinite values.
- Setting warp strength to zero matches the independent/shared-noise baseline.
- Fixed clips show lower temporal flicker while maintaining acceptable box-layout adherence.
- Peak VRAM stays within the agreed single-GPU budget.

#### Deliverables

- Latent propagation implementation.
- Three-way ablation: independent, shared noise, and latent warp.
- Temporal metrics and visual comparison sheet.
- Failure catalogue covering ghosting, smearing, and identity drift.

#### Exit gate

Latent warping becomes the default only if it improves temporal quality on the fixed benchmark without unacceptable spatial or semantic degradation.

### Phase 6 Temporal Attention Adaptation

**Purpose:** add a stronger learned temporal mechanism if latent warping alone is insufficient.

This phase is conditional. It should not block delivery of a successful latent-warp system.

#### Modules and files

- `ulgf_video/temporal_attention.py`
- `ulgf_video/temporal_adapter.py`
- `train_ULGF_video.py`
- `configs/video/train_temporal_adapter.py`
- `tests/test_temporal_attention.py`
- A targeted hook near the current diffusion loss in `train_UWLGM.py`, or preferably a shared extracted training utility.

#### Work items

1. Select the temporal context: previous frame only, sliding window, or sparse keyframes.
2. Insert temporal attention at explicitly documented U-Net resolutions.
3. Freeze original ULGF weights by default.
4. Implement low-rank adapters compatible with the pinned Diffusers version, or document and validate a controlled dependency upgrade.
5. Add temporal consistency loss using warped latent/features while retaining the original diffusion objective.
6. Save adapters separately from the base checkpoint.

#### Acceptance tests

- Only approved adapter and temporal-module parameters receive gradients.
- A one-frame clip is equivalent to the image baseline within documented tolerance.
- Adapter checkpoints reload without modifying the base checkpoint.
- Training can complete a small overfit experiment on a tiny clip set.
- Temporal loss decreases on the overfit set.
- Full evaluation improves temporal metrics over Phase 5 or provides a documented reason not to adopt the module.
- Training and inference fit the agreed hardware budget.

#### Deliverables

- Optional temporal-attention adapter.
- Training configuration and checkpoint format.
- Overfit test report and full ablation report.
- Go/no-go recommendation comparing complexity with measured benefit.

#### Exit gate

Adopt temporal attention only when its benefit is statistically and visually meaningful relative to latent warping.

### Phase 7 Evaluation and Quality Gates

**Purpose:** evaluate spatial fidelity, temporal stability, identity continuity, diversity, and operational performance.

#### Modules and files

- `ulgf_video/metrics.py`
- `scripts/evaluate_video_clips.py`
- `tests/test_metrics.py`
- `evaluation/README.md`

#### Metric groups

| Area | Suggested measures |
|---|---|
| Contract correctness | Manifest validation rate, missing-file count, invalid-box count, ID/class consistency |
| Layout adherence | Detector-based box IoU or centre error against requested layout |
| Image quality | FID on frames, subject to a fixed preprocessing and reference protocol |
| Temporal consistency | Warped LPIPS, optical-flow-warp error, color-histogram drift, temporal perceptual distance |
| Identity continuity | Feature similarity for the same synthetic instance across frames |
| Diversity | Inter-clip perceptual diversity under different seeds |
| Performance | Seconds per frame, peak VRAM, disk use, failure/retry rate |

#### Acceptance tests

- Metrics return known values on synthetic fixtures.
- Evaluation is deterministic for fixed inputs.
- Every comparison uses the same seed layouts and evaluation preprocessing.
- Metric failures identify the clip and frame rather than silently skipping data.
- Human review uses randomized, blinded side-by-side clips where practical.

#### Deliverables

- Reproducible evaluation command.
- Frozen benchmark seed set.
- Phase 3/4/5/6 comparison report.
- Human-review rubric and completed review sheet.

#### Exit gate

Select the production temporal mode from measured quality, resource cost, and failure behavior—not from implementation novelty.

### Phase 8 Packaging and Release

**Purpose:** make the selected pipeline reproducible and safe for another team member to run.

#### Modules and files

- Final updates to `README.md` and `requirements.txt`.
- `docs/video_generation.md`
- `docs/video_output_schema.md`
- `configs/video/*.py`
- `CHANGELOG.md`
- Example output under a size-appropriate `examples/` directory.

#### Work items

1. Provide commands for environment setup, validation, baseline generation, video generation, resume, and evaluation.
2. Document checkpoint provenance and licensing.
3. Add structured logging and actionable error messages.
4. Add a configuration snapshot to every generated clip.
5. Run a clean-machine or clean-environment installation test.
6. Package only the selected production mode while keeping experimental modes clearly labeled.

#### Acceptance tests

- A new operator can generate and validate a sample clip by following the README.
- No command relies on the original author's absolute paths.
- No command deletes or overwrites user data by default.
- Output validation passes for all release examples.
- Configuration, seed, checkpoint, and code version are recoverable from each output.

#### Deliverables

- Release-ready code and documentation.
- Environment lock file.
- Example clip and manifest.
- Final evaluation report.
- Known-limitations and future-work document.

## 6. Cross-Phase Test Strategy

### Unit tests

Run on CPU and cover schemas, coordinate transforms, motion, deterministic seed derivation, latent warping mathematics, validation, and serialization.

### Integration tests

Use small dimensions, few denoising steps, and short clips. They should verify wiring rather than image quality and should be marked separately from normal unit tests.

### GPU smoke tests

Run on the supported CUDA environment before merging any pipeline or training change. Record peak VRAM and execution time.

### Regression fixtures

Freeze a small set of layouts and expected manifests. Image checks should use checksums only where deterministic kernels are guaranteed; otherwise compare shapes, metadata, and bounded perceptual differences.

### Visual QA

For each milestone, inspect:

- box motion overlays;
- side-by-side consecutive frames;
- looped preview videos;
- failure cases with crowded scenes, small objects, boundary interactions, and overlapping boxes.

## 7. Risk Register

| Risk | Probability | Impact | Mitigation | Trigger or fallback |
|---|---:|---:|---|---|
| Baseline environment cannot be reproduced | High | High | Containerize or lock the original environment; validate one image before refactoring | Pause feature work and isolate dependency/checkpoint problems |
| Old Diffusers version blocks LoRA or attention hooks | High | High | Prototype against pinned version first; keep temporal attention optional | Ship latent warping; evaluate controlled upgrade separately |
| Independent frames flicker severely | Certain | Medium | Preserve as control; add shared conditioning and latent propagation | Expected baseline behavior, not a release blocker by itself |
| Box motion is smooth but implausible | Medium | High | Size-aware speed limits, acceleration limits, collision repulsion, visual QA | Retune motion model before neural work |
| Latent warping causes ghosting or smearing | High | High | Confidence masks, fresh-noise blending, periodic resets | Reduce warp strength or revert to shared-noise mode |
| Object visual identity changes despite stable IDs | High | High | Temporal latent reuse, appearance metrics, optional reference features | Mark IDs as annotation truth while improving generation; do not claim visual consistency prematurely |
| Clip conditioning exceeds CLIP token limit | Medium | High | Split medium/layout text and validate token budget | Deterministically compact layout text or reduce maximum objects |
| Cross-frame attention exceeds 24 GB VRAM | Medium | High | Previous-frame context, attention slicing, checkpointing, adapters | Use latent warping as production mode |
| Annotation and generated pixels diverge | Medium | High | Detector-based layout checks and visual overlays | Reject clips below layout-adherence threshold |
| IDs are lost in YOLO export | High | High | Canonical JSON plus documented tracking sidecar | Never use plain YOLO labels as canonical clip annotations |
| Existing scripts delete or overwrite data | High | High | Remove deletion from defaults; atomic output and explicit overwrite flags | Block release until safety tests pass |
| Metrics improve while videos look worse | Medium | Medium | Blinded human review and multiple complementary metrics | Require both quantitative and visual approval |
| Temporal training data are insufficient | Medium | High | Start training-free; use synthetic motion and self-supervised consistency carefully | Do not adopt learned module without held-out gains |

## 8. Milestones and Suggested Schedule

The schedule is expressed in engineering weeks and assumes one primary developer with intermittent access to a 24 GB GPU.

| Milestone | Phases | Expected duration | Evidence required |
|---|---|---:|---|
| M0 Reproducible baseline | Phase 0 | 1-2 weeks | Safe command, baseline samples, environment report |
| M1 Valid clip layouts | Phases 1-2 | 2 weeks | Schema, validator, motion previews, tests |
| M2 End-to-end video MVP | Phase 3 | 1-2 weeks | Generated clips, manifests, baseline metrics |
| M3 Stable clip appearance | Phase 4 | 1 week | Conditioning ablation and updated clips |
| M4 Temporal prototype | Phase 5 | 2-3 weeks | Latent-warp ablation, metrics, VRAM report |
| M5 Learned temporal option | Phase 6 | 3-5 weeks, optional | Adapter checkpoint and go/no-go report |
| M6 Release candidate | Phases 7-8 | 1-2 weeks | Evaluation report, clean setup test, documentation |

## 9. Decision Gates

1. **Baseline gate:** Can stock-style ULGF inference be reproduced safely?
2. **Motion gate:** Are layout sequences valid and visually plausible without diffusion?
3. **MVP gate:** Can the system generate, resume, export, and validate an independent-frame clip?
4. **Conditioning gate:** Does shared medium conditioning improve clip consistency without hurting layout adherence?
5. **Warp gate:** Does latent propagation outperform the independent baseline at acceptable cost?
6. **Attention gate:** Is a learned temporal adapter worth its complexity and hardware cost?
7. **Release gate:** Can a new operator reproduce a validated clip from documented commands?

Each gate requires stored evidence. A phase is not complete merely because its code runs.

## 10. Immediate Backlog

Complete these tasks first, in order:

1. Add a safe baseline CLI with no hardcoded dataset, model, seed-file, or output paths.
2. Make deletion and overwrite explicit opt-in operations.
3. Fix the training launcher process/GPU configuration.
4. Create the versioned clip schema and validator.
5. Add deterministic instance-ID assignment.
6. Implement and visualize a constant-velocity sequence generator.
7. Freeze a small benchmark seed set.
8. Generate the first independent-frame four-frame clip.

These tasks establish the foundation for every temporal-model experiment and should be completed before modifying U-Net attention.
