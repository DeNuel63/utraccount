"""COVTrack inference on external original-image xyxy detections.

The detector's RPN, classification and box regression do not select or replace
these detections. Its pretrained RoI feature heads and association remain in use.
"""
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import bbox2result, bbox2roi
from ..core import track2result


def extract_external_features(head, features, boxes, metadata):
    if getattr(head, 'init_track_head_by_bbox_head', False) and not head.init_track_head_tag:
        head.init_track_head_by_bbox_head_method()
    track_boxes = boxes[:, :4] * boxes.new_tensor(metadata['scale_factor'])
    appearance = head._track_forward(features, [track_boxes])
    cem = head._cem_forward(features, [track_boxes]) if head.cem_head is not None else None
    semantic = None
    if getattr(head, 'use_cls_static_ratio', False) and getattr(head, 'use_motion_static_ratio', False):
        rois = bbox2roi([track_boxes])
        _, text = head._bbox_forward(features, rois)
        text = F.normalize(head.projection(text), p=2, dim=1)
        if head.ensemble:
            _, image = head._bbox_forward_for_image(features, rois)
            image = F.normalize(head.projection_for_image(image), p=2, dim=1)
            semantic = (text + image) / 2
        else:
            semantic = text
        if hasattr(head, 'loc_head') and hasattr(head, 'cls_head') and hasattr(head, 'stog'):
            # Match the existing head's trained location normalization and MAN.
            from ..models.roi_heads.ovtrack_roi_head import normalize_detections
            fused = appearance + head.cls_head(semantic) + head.loc_head(normalize_detections(boxes))
            appearance, _ = head.stog(fused.unsqueeze(0), fused.unsqueeze(0))
            appearance = appearance.squeeze(0)
    return appearance, semantic if semantic is not None else (cem if cem is not None else appearance)


def track_external_detections(model, image, metadata, detections):
    if image.size(0) != 1 or len(metadata) != 1:
        raise ValueError('External tracking accepts one frame at a time')
    meta = metadata[0]
    if meta.get('flip', False):
        raise ValueError('External tracking requires an unflipped test pipeline')
    frame_id = detections['frame_index']
    sequence_id = detections['sequence_id']
    if frame_id == 0:
        model.init_tracker()
        model._external_sequence_id = sequence_id
        model._external_next_frame = 0
        head = model.roi_head
        if getattr(head, 'use_cls_static_ratio', False) and getattr(head, 'use_motion_static_ratio', False):
            model.tracker.set_fusion_head(head.fusion_head, head.track_head.loss_cyc)
    if sequence_id != getattr(model, '_external_sequence_id', None) or frame_id != getattr(model, '_external_next_frame', None):
        raise ValueError('External frames must be consecutive within one sequence, starting at zero')
    meta['frame_id'] = frame_id
    h, w = meta['ori_shape'][:2]
    if (w, h) != (detections['width_px'], detections['height_px']):
        raise ValueError('External frame dimensions do not match preprocessing')
    device = image.device
    boxes = torch.as_tensor(detections['boxes'], dtype=torch.float32, device=device).reshape(-1, 5)
    labels = torch.as_tensor(detections['labels'], dtype=torch.long, device=device)
    if len(labels) != len(boxes) or not torch.isfinite(boxes).all():
        raise ValueError('Invalid external boxes or labels')
    if len(boxes) and ((boxes[:, 2:4] <= boxes[:, :2]).any() or (boxes[:, :2] < 0).any()
        or (boxes[:, 2] > w).any() or (boxes[:, 3] > h).any()
        or (boxes[:, 4] < 0).any() or (boxes[:, 4] > 1).any()
        or (labels < 0).any() or (labels >= model.roi_head.num_classes).any()):
        raise ValueError('External detections violate image/class bounds')
    if len(boxes):
        features = model.extract_feat(image)
        appearance, semantic = extract_external_features(model.roi_head, features, boxes, meta)
        tracked, track_labels, ids = model.tracker.match(bboxes=boxes, labels=labels,
            embeds=appearance, cls_embeds=semantic, frame_id=frame_id, method=model.method,
            filename=meta.get('filename') or 'custom_video')
        tracks = track2result(tracked, track_labels, ids, model.roi_head.num_classes)
    else:
        # Native empty-frame behavior: retain memory, advance frame order.
        tracks = [np.zeros((0, 6), dtype=np.float32) for _ in range(model.roi_head.num_classes)]
    model._external_next_frame += 1
    return dict(bbox_results=bbox2result(boxes, labels, model.roi_head.num_classes), track_results=tracks)
