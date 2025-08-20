import os
import json
import random
from glob import glob
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from transformers import pipeline
from torchvision.ops import nms
import shutil

from color_utils import compute_dominant_color_id

# Paths
IMAGES_DIR = "/Users/tomas.komar/digitecus/fotos/processed/images"
DATASET_DIR = "/Users/tomas.komar/digitecus/fotos/dataset"
IM_TRAIN = os.path.join(DATASET_DIR, "images", "train")
IM_VAL = os.path.join(DATASET_DIR, "images", "val")
LBL_TRAIN = os.path.join(DATASET_DIR, "labels", "train")
LBL_VAL = os.path.join(DATASET_DIR, "labels", "val")

# Classes (index must match dataset.yaml). We include 'any/unknown' as class 0 but do not use it for training labels.
CLASSES = [
	"any/unknown",
	"can",
	"duck",
	"cup",
	"sponge",
	"ball",
	"vegetable",
]

# Candidate labels for zero-shot detector (exclude unknown)
ZS_LABELS = [
	"can", "soda can", "tin can",
	"rubber duck", "duck",
	"cup", "mug",
	"sponge", "kitchen sponge",
	"ball",
	"vegetable", "carrot", "tomato", "cucumber",
]
ZS_TO_CLASS = {
	"can": 1,
	"soda can": 1,
	"tin can": 1,
	"rubber duck": 2,
	"duck": 2,
	"cup": 3,
	"mug": 3,
	"sponge": 4,
	"kitchen sponge": 4,
	"ball": 5,
	"vegetable": 6,
	"carrot": 6,
	"tomato": 6,
	"cucumber": 6,
}

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
VAL_RATIO = 0.1
RANDOM_SEED = 42


def ensure_dirs():
	for d in [IM_TRAIN, IM_VAL, LBL_TRAIN, LBL_VAL]:
		os.makedirs(d, exist_ok=True)


def run_detector() -> pipeline:
	device = 0 if torch.cuda.is_available() else -1
	return pipeline(
		"zero-shot-object-detection",
		model="google/owlvit-base-patch16",
		device=device,
	)


def to_yolo_bbox(box: Dict[str, float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
	xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
	x_c = (xmin + xmax) / 2.0 / img_w
	y_c = (ymin + ymax) / 2.0 / img_h
	w = (xmax - xmin) / img_w
	h = (ymax - ymin) / img_h
	return x_c, y_c, w, h


def to_xyxy(box: Dict[str, float]) -> Tuple[int, int, int, int]:
	return int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])


def classwise_nms(dets: List[Dict], iou_thr: float) -> List[Dict]:
	if not dets:
		return []
	# group by class id
	by_c: Dict[int, List[int]] = {}
	for i, d in enumerate(dets):
		by_c.setdefault(d["class_id"], []).append(i)
	keep_indices = []
	for cid, idxs in by_c.items():
		boxes = torch.tensor([dets[i]["xyxy"] for i in idxs], dtype=torch.float32)
		scores = torch.tensor([dets[i]["score"] for i in idxs], dtype=torch.float32)
		kept_local = nms(boxes, scores, iou_thr).tolist()
		keep_indices.extend([idxs[k] for k in kept_local])
	return [dets[i] for i in keep_indices]


def label_image(detector: pipeline, image_path: str) -> Tuple[List[str], List[int]]:
	img = cv2.imread(image_path, cv2.IMREAD_COLOR)
	if img is None:
		return [], []
	h, w = img.shape[:2]
	results = detector(image_path, candidate_labels=ZS_LABELS)
	# Normalize and filter
	detections = []
	for r in results:
		label = r["label"]
		score = float(r["score"])
		if score < CONF_THRESHOLD:
			continue
		class_id = ZS_TO_CLASS.get(label, None)
		if class_id is None:
			continue
		box = r["box"]
		xyxy = to_xyxy(box)
		color_id = compute_dominant_color_id(img, xyxy)
		detections.append({
			"class_id": class_id,
			"score": score,
			"xyxy": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
			"colors_id": color_id,
			"yolo": to_yolo_bbox(box, w, h),
		})
	# NMS
	detections = classwise_nms(detections, IOU_THRESHOLD)
	# Serialize YOLO label lines and colors list (in same order)
	lines = []
	colors = []
	for d in detections:
		cx, cy, bw, bh = d["yolo"]
		lines.append(f"{d['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
		colors.append(int(d["colors_id"]))
	return lines, colors


def split_train_val(files: List[str]) -> Tuple[List[str], List[str]]:
	random.Random(RANDOM_SEED).shuffle(files)
	n_val = max(1, int(len(files) * VAL_RATIO))
	val_files = files[:n_val]
	train_files = files[n_val:]
	return train_files, val_files


def safe_copy(src: str, dst: str) -> None:
	try:
		os.link(src, dst)
	except Exception:
		shutil.copy2(src, dst)


def main():
	ensure_dirs()
	det = run_detector()
	image_files = sorted(glob(os.path.join(IMAGES_DIR, "*.png")))
	train_files, val_files = split_train_val(image_files)
	for split_name, files, im_out, lbl_out in [
		("train", train_files, IM_TRAIN, LBL_TRAIN),
		("val", val_files, IM_VAL, LBL_VAL),
	]:
		for img_path in files:
			lines, colors = label_image(det, img_path)
			# copy image
			basename = os.path.basename(img_path)
			out_img_path = os.path.join(im_out, basename)
			if not os.path.exists(out_img_path):
				safe_copy(img_path, out_img_path)
			# write labels if any
			stem = os.path.splitext(basename)[0]
			lbl_path = os.path.join(lbl_out, f"{stem}.txt")
			col_path = os.path.join(lbl_out, f"{stem}.colors.json")
			with open(lbl_path, "w", encoding="utf-8") as f:
				f.write("\n".join(lines))
			with open(col_path, "w", encoding="utf-8") as f:
				json.dump({"colors": colors}, f)
	print(f"Auto-labeling complete. Dataset in {DATASET_DIR}")


if __name__ == "__main__":
	main()
