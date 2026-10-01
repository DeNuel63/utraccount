# ULGF training-free video experiment — final diagnostic handoff

Date: 2026-09-30

## Decision and scope

This branch is complete as a **single-object, training-free compositing diagnostic**.
It is not completed ULGF training, a learned temporal model, realistic swimming,
or a production-ready synthetic tracking dataset. No optimizer updates were performed
in this branch. Normal ULGF training preparation remains a separate workstream;
this handoff does not clear its outstanding readiness gates.

The accepted composition combines a RUOD-derived, low-strength image-conditioned
fish, a user-drawn and user-approved binary silhouette, and a user-supplied clean
background plate. Frames copy the selected fish pixels at translated coordinates.
The pipeline assigns persistent ID 1; it does not discover or track that identity.

## Evidence and progression

1. Independent ULGF calls with moving layouts changed fish appearance. Repeating
   identical inputs produced identical pixels in the tested zero-motion control.
2. An isolated scheduling correction restored the first five omitted scheduler
   calls in the no-prior path. A matched two-seed comparison recorded 96 versus
   101 UNet calls, without resolving the observed anatomy problems.
3. Source-image conditioning at strengths 0.15, 0.25 and 0.35 retained broad scene
   characteristics but damaged the small fish. A float32/float16 decoder mismatch
   was corrected in the experimental runner; no frozen baseline edits were made.
4. A square 813x813 fish-centred crop, uniformly resized to 256x256, removed
   letterbox padding. Strength 0.05 was selected for mask review. This remains
   source-derived imagery, not an independently generated new animal.
5. A user-guided v3 mask replaced earlier inaccurate outlines. The user confirmed:
   "The cyan boundary follows my intended outline". The mask contains 4,932 pixels.
6. Boundary-colour propagation and exemplar patch filling left visible artifacts.
   The user supplied a clean plate; it was centre-cropped and uniformly resized.
   The user accepted the resulting short composition and playback.
7. Both a five-frame sequence and a separate 48-frame translation trial were
   validated and packaged. The longer trial has structural/pixel verification and
   assistant frame/timing review, not a separately recorded user visual approval.

## Delivered sequences

| Property | sequence_5frame | sequence_48frame |
| --- | --- | --- |
| Frames | 5 | 48 |
| Canvas | 256x256 RGB | 256x256 RGB |
| Playback | 4 fps, 1.25 seconds | 10 fps, 4.8 seconds |
| Motion | (0,0), (5,3), (9,6), (14,9), (19,11) pixels | cosine-eased integer translation, (0,0) to (40,20) |
| Persistent identity | ID 1, fish, class index 4 | ID 1, fish, class index 4 |
| Visual status | user-accepted diagnostic | assistant reviewed; user review pending |

The five-frame packaged playback has uniform timing. The earlier archived review
preview included an extra final-frame hold; it is not the timing contract here.
All playback loops jump back to frame zero. This reset is not continuous swimming.
The long clip has 38 distinct integer positions and 10 stationary transitions;
lossless WebP may merge repeated images while preserving all 4,800 milliseconds.

Each sequence contains PNG frames, binary masks, per-frame annotations,
`manifest.json`, lossless `playback.webp`, a contact sheet, provenance,
`validation.json`, and `SHA256SUMS.json`. No MP4 is claimed or included.

## Validation results

- All 53 frame annotations exactly bound their corresponding translated masks.
- Normalized boxes match pixel boxes; pixel maxima are exclusive.
- Every frame has `instance_id: 1`, `class_id: 4`, `class_name: fish`.
- All selected foreground pixels are preserved exactly at their new coordinates.
- Every unmasked pixel equals the same static plate, including frame zero.
- No mask clipping occurs. Canonical schema validation passed for both sequences.
- Lossless playback was decoded and checked against frame pixels and timing.
- Per-sequence archives and file hashes were checked; the accepted source run was
  verified unchanged. Six packaging/validation regression tests passed.

These checks establish internal consistency, not independent biological realism,
annotation ground-truth accuracy, tracking performance, or distributional quality.

## Identity, provenance and evaluation boundaries

The keyframe derives from RUOD `simple_012339.jpg`. Its conditioned-image SHA-256 is
`8d0974e0f6bad3039970ceafb85637d71e710462259039a3c6580d8e6a1f76a1`.
The approved decoded-mask SHA-256 is
`aee93c2fe59758692cefe1615fac9036c27bfca4d6cfa27e76413b1d4ee1bf14`.
Full source/checkpoint hashes and configuration accompany the image-conditioning
report in each sequence's provenance. Weights are not distributed in this bundle.

The plate is a supplied replacement, not a measurement of the background hidden
by the fish. The binary fin pixels still contain mixtures of original background
colour. No exact alpha matting, occlusion/depth handling, fin articulation, camera
motion, multi-object interaction, or learned temporal consistency is implemented.
The fish appears softer than the plate. Integer motion can produce slight stepping.

This one-source diagnostic must not be used as independent holdout evidence.
Review image/checkpoint provenance and redistribution rights before external or
public distribution; the user's supplied plate is not automatically licensed for
redistribution. No new upstream license or dataset license is asserted here.

## How to use the handoff

1. Extract the handoff ZIP into a new directory; do not overwrite existing runs.
2. In a CPU Python environment with Pillow and NumPy, run:
   `python verify_delivery.py .`
3. Inspect each `playback.webp` and `contact_sheet.png`.
4. Use `manifest.json` for normalized xyxy boxes and persistent IDs; use the
   per-frame annotation files for pixel coordinates and explicit provenance.

Validation requires no checkpoint, CUDA, Torch, MMCV or MMDetection. The bundle is
self-contained for playback, checking annotations and reproducing the rigid
composition from its included keyframe, mask, plate and offsets. It does not
provide the environment/weights needed to regenerate the original conditioned fish.

## Closure checklist and next workstream

- [x] Keyframe, user-guided mask, supplied plate and short playback reviewed.
- [x] Five-frame annotations, masks, IDs, provenance and playback packaged.
- [x] Longer bounded trajectory generated and checked.
- [x] Limitations and the separation from training explicitly recorded.
- [ ] Optional: user visual sign-off on the longer trajectory.
- [ ] Separate future scope: natural motion, soft alpha edges, multiple objects,
  occlusion, broader source diversity and independent quantitative evaluation.
- [ ] Separate normal-training scope: current leakage clearance, initialization
  provenance, pinned GPU environment, real update/VRAM/resume tests and pilot review.

Do not infer completion of normal training from this diagnostic handoff.
