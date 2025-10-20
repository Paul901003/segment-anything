#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SAM on COCO images.

Modes:
  - auto  : Fully automatic masks via SamAutomaticMaskGenerator
  - annot : Use COCO annotations (bboxes) as prompts for SamPredictor

Outputs (per image):
  - <out_dir>/<split>/vis/<image_id>.png         (visualization)
  - <out_dir>/<split>/json/<image_id>.jsonl      (one JSON object per mask)

Requirements:
  pip install "pycocotools>=2.0.7" opencv-python numpy torch torchvision
  pip install -e .   # at the segment-anything repo root

Example:
  python run_coco_with_sam.py \
    --coco-root /path/to/coco \
    --split val2017 \
    --ann-file /path/to/coco/annotations/instances_val2017.json \
    --checkpoint weights/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --mode auto \
    --out-dir out_coco_sam \
    --max-images 100
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

from pycocotools.coco import COCO

import torch

# SAM imports (after pip install -e . at repo root)
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Return RGB color for a binary mask (random but deterministic per call)."""
    # Random bright color
    color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(3):
        colored[:, :, c] = np.where(mask, color[c], 0)
    return colored


def overlay_masks_on_image(img_bgr: np.ndarray, masks: List[Dict[str, Any]], alpha: float = 0.6) -> np.ndarray:
    vis = img_bgr.copy()
    overlay = np.zeros_like(img_bgr)
    for m in masks:
        mask = m["segmentation"]  # binary np.array HxW
        colored = colorize_mask(mask)
        overlay = np.where(mask[..., None], colored, overlay)
    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    return vis


def rle_encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    """
    COCO-style RLE using pycocotools' mask encoding; expects Fortran contiguous.
    """
    from pycocotools import mask as maskUtils

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    rle = maskUtils.encode(np.asfortranarray(mask))
    # pycocotools returns counts as bytes; convert to str for JSON
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def sam_auto_generate(sam, image_rgb: np.ndarray, amg_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    generator = SamAutomaticMaskGenerator(sam, **(amg_kwargs or {}))
    # returns list of dict with keys: segmentation (np.bool_), area, bbox, predicted_iou, stability_score, point_coords, etc.
    masks = generator.generate(image_rgb)
    # normalize output to the format we dump later
    norm = []
    for m in masks:
        norm.append(
            dict(
                segmentation=m["segmentation"].astype(np.uint8),  # HxW
                area=int(m["area"]),
                bbox=[float(x) for x in m["bbox"]],
                score=float(m.get("predicted_iou", 0.0)),
            )
        )
    return norm


def sam_from_bboxes(predictor: SamPredictor, image_rgb: np.ndarray, bboxes_xywh: List[List[float]]) -> List[Dict[str, Any]]:
    """
    Use COCO bboxes as prompts. predictor.transform.apply_boxes_torch will map to model input space.
    """
    predictor.set_image(image_rgb)
    device = predictor.model.device
    import torch

    if len(bboxes_xywh) == 0:
        return []

    # Convert xywh -> xyxy
    xyxy = []
    H, W = image_rgb.shape[:2]
    for (x, y, w, h) in bboxes_xywh:
        x2 = x + w
        y2 = y + h
        # clamp within image
        x, y = max(0, x), max(0, y)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 <= x or y2 <= y:
            continue
        xyxy.append([x, y, x2, y2])

    if len(xyxy) == 0:
        return []

    boxes = torch.tensor(xyxy, dtype=torch.float, device=device)
    transformed = predictor.transform.apply_boxes_torch(boxes, image_rgb.shape[:2])

    masks, scores, logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed,
        multimask_output=False,  # one mask per box
    )

    masks_np = masks.detach().cpu().numpy()  # (N,1,H,W) or (N,H,W) depending on SAM version
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0, :, :]
    scores_np = scores.detach().cpu().numpy().reshape(-1)

    results = []
    for i, m in enumerate(masks_np):
        # ensure binary uint8
        m_bin = (m > 0.0).astype(np.uint8)
        x1, y1, x2, y2 = xyxy[i]
        w = x2 - x1
        h = y2 - y1
        results.append(
            dict(
                segmentation=m_bin,
                area=int(m_bin.sum()),
                bbox=[float(x1), float(y1), float(w), float(h)],
                score=float(scores_np[i]),
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Run SAM on COCO images (auto or annot mode).")
    parser.add_argument("--coco-root", type=str, required=True, help="Path to COCO root folder (contains images/<split>)")
    parser.add_argument("--split", type=str, default="val2017", choices=["train2017", "val2017"], help="COCO split folder name")
    parser.add_argument("--ann-file", type=str, required=True, help="Path to COCO instances_*.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM checkpoint .pth")
    parser.add_argument("--model-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "annot"], help="auto=automatic masks; annot=use COCO bboxes")
    parser.add_argument("--out-dir", type=str, default="out_coco_sam", help="Output directory")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--amg-params", type=str, default="", help="JSON string of kwargs for SamAutomaticMaskGenerator")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    split_dir = coco_root / "images" / args.split
    if not split_dir.exists():
        # Some setups put images directly under /val2017 instead of /images/val2017
        alt = coco_root / args.split
        if alt.exists():
            split_dir = alt
        else:
            raise FileNotFoundError(f"Cannot find images dir: {split_dir} or {alt}")

    out_base = Path(args.out_dir) / args.split
    vis_dir = out_base / "vis"
    json_dir = out_base / "json"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO
    coco = COCO(args.ann_file)
    img_ids = coco.getImgIds()
    if args.max_images and args.max_images > 0:
        img_ids = img_ids[: args.max_images]

    # Build SAM
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)

    amg_kwargs = {}
    if args.amg_params:
        try:
            amg_kwargs = json.loads(args.amg_params)
        except Exception as e:
            print(f"Failed to parse --amg-params JSON: {e}")

    predictor = SamPredictor(sam) if args.mode == "annot" else None

    print(f"Total images: {len(img_ids)} | split={args.split} | mode={args.mode} | device={args.device}")

    for idx, img_id in enumerate(img_ids):
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        img_path = str(split_dir / file_name)

        # BGR -> RGB
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if args.mode == "auto":
            masks = sam_auto_generate(sam, img_rgb, amg_kwargs)
        else:
            # Use COCO bboxes as prompts
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            bboxes_xywh = [a["bbox"] for a in anns if "bbox" in a and a.get("iscrowd", 0) == 0]
            masks = sam_from_bboxes(predictor, img_rgb, bboxes_xywh)

        # Dump JSONL and visualization
        jsonl_path = json_dir / f"{img_id}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for m in masks:
                rle = rle_encode_binary_mask(m["segmentation"])
                # store minimal fields for eval/analysis
                record = dict(
                    image_id=int(img_id),
                    bbox=[float(x) for x in m["bbox"]],
                    area=int(m["area"]),
                    score=float(m.get("score", 0.0)),
                    segmentation=rle,  # COCO-style RLE
                )
                f.write(json.dumps(record) + "\n")

        # Visualization
        vis = overlay_masks_on_image(img_bgr, masks)
        vis_path = vis_dir / f"{img_id}.png"
        cv2.imwrite(str(vis_path), vis)

        if (idx + 1) % 20 == 0:
            print(f"[{idx + 1}/{len(img_ids)}] saved -> {vis_path.name}, {jsonl_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
