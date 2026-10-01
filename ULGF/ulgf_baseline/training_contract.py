"""Training contracts shared by the RUOD dataset and checkpoint exporter."""
import math
import random

import numpy as np

RUOD_PROMPT_TEMPLATE = "A underwater scene image of {camera} camera with {bbox}"


def checked_token_ids(tokenizer, captions):
    ids = tokenizer(captions, padding=False, truncation=False).input_ids
    for index, tokens in enumerate(ids):
        if len(tokens) > tokenizer.model_max_length:
            raise ValueError("Prompt {} has {} tokens; limit {}. Refusing silent box truncation.".format(
                index, len(tokens), tokenizer.model_max_length))
    return ids


def make_loss_mask(boxes, height, width, mode, weight, normalize, rng=None):
    """Random ownership per overlap region; legacy 'constant' uses area^-0.2.

    Geometry is normalized XYXY. Tiny valid boxes cover at least one latent
    cell. Object ownership and floating point weights are separate arrays.
    """
    if height <= 0 or width <= 0 or mode not in (None, "constant", "area"):
        raise ValueError("Invalid mask dimensions or loss mode")
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("Foreground loss weight must be finite and positive")
    rng = rng or random
    regions = {}
    weights = []
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = map(float, box)
        if not all(math.isfinite(v) for v in box) or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Invalid normalized box: {}".format(box))
        left = min(width - 1, int(round(x1 * width)))
        top = min(height - 1, int(round(y1 * height)))
        right = min(width, max(left + 1, int(round(x2 * width))))
        bottom = min(height, max(top + 1, int(round(y2 * height))))
        area = (right - left) * (bottom - top)
        weights.append(weight * area ** -0.2 if mode == "constant" else area ** -weight)
        for y in range(top, bottom):
            for x in range(left, right):
                regions.setdefault((y, x), []).append(index)
    if mode is None:
        return np.ones((height, width), dtype=np.float32)
    background = (height * width) ** (-0.2 if mode == "constant" else -weight)
    mask = np.full((height, width), background, dtype=np.float32)
    owners = {}
    for point, candidates in regions.items():
        key = tuple(candidates)
        if key not in owners:
            owners[key] = candidates[0] if len(candidates) == 1 else rng.choice(candidates)
        mask[point] = weights[owners[key]]
    if normalize:
        mask /= mask.mean()
    if not np.isfinite(mask).all() or (mask <= 0).any():
        raise ValueError("Loss mask contains nonfinite or nonpositive values")
    return mask


def generation_config(dataset, buckets, image_size):
    height, width = image_size
    return dict(dataset=dataset, num_bucket_per_side=list(buckets),
                width=int(width), height=int(height),
                prompt_template=RUOD_PROMPT_TEMPLATE, cfg_scale=5.0,
                num_inference_steps=100, max_num_bbox=22)
