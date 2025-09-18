import os
import json
import random
from glob import glob
from typing import List, Dict, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import cv2
import numpy as np
import torch
from transformers import pipeline
from torchvision.ops import nms
import shutil

from color_utils import compute_dominant_color_id

# Paths
IMAGES_DIR = "/home/ut-ai/ai-works/adaptibot/yolo_detect/processed/images"
DATASET_DIR = "/home/ut-ai/ai-works/adaptibot/yolo_detect/dataset"
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
	"can", "soda can", "tin can", "aluminum can", "drink can", "beverage can",
	"rubber duck", "duck", "toy duck", "yellow duck", "bath duck",
	"cup", "mug", "plastic cup", "paper cup", "coffee cup", "tea cup", "drinking cup", "glass", "tumbler",
	"sponge", "kitchen sponge", "dish sponge", "cleaning sponge", "scrub sponge", "wash sponge",
	"ball", "tennis ball", "rubber ball", "toy ball", "small ball", "round ball",
	"vegetable", "carrot", "tomato", "cucumber", "pepper", "bell pepper", "green pepper", "red pepper", "orange", "fruit",
]
ZS_TO_CLASS = {
	"can": 1, "soda can": 1, "tin can": 1, "aluminum can": 1, "drink can": 1, "beverage can": 1,
	"rubber duck": 2, "duck": 2, "toy duck": 2, "yellow duck": 2, "bath duck": 2,
	"cup": 3, "mug": 3, "plastic cup": 3, "paper cup": 3, "coffee cup": 3, "tea cup": 3, "drinking cup": 3, "glass": 3, "tumbler": 3,
	"sponge": 4, "kitchen sponge": 4, "dish sponge": 4, "cleaning sponge": 4, "scrub sponge": 4, "wash sponge": 4,
	"ball": 5, "tennis ball": 5, "rubber ball": 5, "toy ball": 5, "small ball": 5, "round ball": 5,
	"vegetable": 6, "carrot": 6, "tomato": 6, "cucumber": 6, "pepper": 6, "bell pepper": 6, "green pepper": 6, "red pepper": 6, "orange": 6, "fruit": 6,
}

CONF_THRESHOLD = 0.3
CLASS_MIN_CONF = {
	1: 0.40,  # can - moderate to reduce dominance
	2: 0.20,  # duck - very low to catch more
	3: 0.20,  # cup - very low to catch more
	4: 0.25,  # sponge - low
	5: 0.20,  # ball - very low to catch more
	6: 0.20,  # vegetable - very low to catch more
}
IOU_THRESHOLD = 0.7  # Lower NMS threshold to keep more overlapping detections
CROSS_CLASS_OVERLAP_THRESHOLD = 0.6  # Remove overlapping bboxes from different classes above this IoU
VAL_RATIO = 0.1
RANDOM_SEED = 42

# Multi-GPU configuration
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
BATCH_SIZE_PER_GPU = 4  # Number of images to process per batch on each GPU
MAX_WORKERS = NUM_GPUS  # One worker per GPU


def get_optimal_batch_size() -> int:
	"""Determine optimal batch size based on GPU memory"""
	if not torch.cuda.is_available():
		return 1
	
	# Get GPU memory in GB
	gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
	
	# Estimate batch size based on GPU memory (OWL-ViT large needs ~2-3GB per image)
	if gpu_memory_gb >= 24:  # RTX 4090, A6000, etc.
		return 6
	elif gpu_memory_gb >= 16:  # RTX 4080, etc.
		return 4
	elif gpu_memory_gb >= 12:  # RTX 4070 Ti, etc.
		return 3
	elif gpu_memory_gb >= 8:   # RTX 4060 Ti, etc.
		return 2
	else:
		return 1


def ensure_dirs():
	for d in [IM_TRAIN, IM_VAL, LBL_TRAIN, LBL_VAL]:
		os.makedirs(d, exist_ok=True)


def run_detector(gpu_id: int = 0) -> pipeline:
	"""Create a detector pipeline for a specific GPU"""
	if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
		device = gpu_id
		print(f"Worker GPU {gpu_id}: Using {torch.cuda.get_device_name(gpu_id)}")
	else:
		device = -1
		print(f"Worker {gpu_id}: CUDA not available, using CPU")
	
	return pipeline(
		"zero-shot-object-detection",
		model="google/owlvit-large-patch14",
		device=device,
		torch_dtype=torch.float16 if device >= 0 else torch.float32,
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


def calculate_iou(box1: List[float], box2: List[float]) -> float:
	"""Calculate Intersection over Union (IoU) between two bounding boxes in xyxy format"""
	x1_min, y1_min, x1_max, y1_max = box1
	x2_min, y2_min, x2_max, y2_max = box2
	
	# Calculate intersection
	inter_x_min = max(x1_min, x2_min)
	inter_y_min = max(y1_min, y2_min)
	inter_x_max = min(x1_max, x2_max)
	inter_y_max = min(y1_max, y2_max)
	
	if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
		return 0.0
	
	inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
	
	# Calculate union
	box1_area = (x1_max - x1_min) * (y1_max - y1_min)
	box2_area = (x2_max - x2_min) * (y2_max - y2_min)
	union_area = box1_area + box2_area - inter_area
	
	return inter_area / union_area if union_area > 0 else 0.0


def filter_cross_class_overlaps(dets: List[Dict], overlap_threshold: float = 0.75) -> List[Dict]:
	"""Remove overlapping bboxes from different classes, keeping the one with higher confidence"""
	if len(dets) <= 1:
		return dets
	
	# Sort by confidence (highest first)
	sorted_dets = sorted(dets, key=lambda x: x["score"], reverse=True)
	keep = []
	
	for i, det1 in enumerate(sorted_dets):
		should_keep = True
		
		for j, det2 in enumerate(keep):
			# Only check overlap between different classes
			if det1["class_id"] != det2["class_id"]:
				iou = calculate_iou(det1["xyxy"], det2["xyxy"])
				if iou >= overlap_threshold:
					# det2 has higher confidence (already in keep list), so skip det1
					should_keep = False
					break
		
		if should_keep:
			keep.append(det1)
	
	return keep


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
	
	# Debug: show all detections above 0.05 confidence
	img_name = os.path.basename(image_path)
	all_dets = [r for r in results if r["score"] >= 0.05]
	if all_dets:
		print(f"{img_name}: Found {len(all_dets)} raw detections:")
		for r in all_dets[:15]:  # Show top 15
			print(f"  {r['label']}: {r['score']:.3f}")
	else:
		print(f"{img_name}: No detections above 0.05 confidence!")
	
	# Normalize and filter
	detections = []
	for r in results:
		label = r["label"]
		score = float(r["score"])
		class_id = ZS_TO_CLASS.get(label, None)
		if class_id is None:
			continue
		min_conf = CLASS_MIN_CONF.get(class_id, CONF_THRESHOLD)
		if score < min_conf:
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
	
	# Debug: show final detections by class
	if detections:
		class_counts = {}
		for d in detections:
			class_counts[d["class_id"]] = class_counts.get(d["class_id"], 0) + 1
		print(f"  Before NMS: {len(detections)} detections, {class_counts}")
	else:
		print(f"  No detections passed confidence filtering!")
	
	# Apply class-wise NMS first
	detections = classwise_nms(detections, IOU_THRESHOLD)
	
	# Debug: show detections after class-wise NMS
	if detections:
		class_counts_nms = {}
		for d in detections:
			class_counts_nms[d["class_id"]] = class_counts_nms.get(d["class_id"], 0) + 1
		print(f"  After class-wise NMS: {len(detections)} detections, {class_counts_nms}")
	
	# Apply cross-class overlap filtering
	detections = filter_cross_class_overlaps(detections, overlap_threshold=CROSS_CLASS_OVERLAP_THRESHOLD)
	
	# Debug: show final detections after cross-class filtering
	if detections:
		class_counts_final = {}
		for d in detections:
			class_counts_final[d["class_id"]] = class_counts_final.get(d["class_id"], 0) + 1
		print(f"  After cross-class filtering: {len(detections)} detections, {class_counts_final}")
	else:
		print(f"  No detections remaining after filtering!")
	
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


def process_images_batch(gpu_id: int, image_batch: List[str]) -> List[Tuple[str, List[str], List[int]]]:
	"""Process a batch of images on a specific GPU"""
	print(f"GPU {gpu_id}: Processing batch of {len(image_batch)} images")
	
	# Initialize detector for this GPU
	detector = run_detector(gpu_id)
	
	results = []
	for img_path in image_batch:
		try:
			lines, colors = label_image(detector, img_path)
			results.append((img_path, lines, colors))
		except Exception as e:
			print(f"GPU {gpu_id}: Error processing {os.path.basename(img_path)}: {e}")
			results.append((img_path, [], []))
	
	print(f"GPU {gpu_id}: Completed batch processing")
	return results


def create_batches(files: List[str], batch_size: int) -> List[List[str]]:
	"""Split files into batches for parallel processing"""
	batches = []
	for i in range(0, len(files), batch_size):
		batches.append(files[i:i + batch_size])
	return batches


def process_split_parallel(split_name: str, files: List[str], im_out: str, lbl_out: str) -> None:
	"""Process a train/val split using parallel GPU workers"""
	if not files:
		return
	
	print(f"\n🚀 Processing {split_name} split: {len(files)} images using {NUM_GPUS} GPUs")
	start_time = time.time()
	
	# Create batches for parallel processing
	total_batch_size = BATCH_SIZE_PER_GPU * NUM_GPUS
	batches = create_batches(files, total_batch_size)
	
	print(f"Created {len(batches)} batches (batch size: {total_batch_size})")
	
	all_results = []
	completed_images = 0
	
	# Process batches in parallel
	with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
		# Submit jobs for parallel processing
		future_to_batch = {}
		
		for batch_idx, batch in enumerate(batches):
			# Distribute batch across GPUs
			gpu_batches = create_batches(batch, BATCH_SIZE_PER_GPU)
			
			for gpu_id, gpu_batch in enumerate(gpu_batches):
				if gpu_batch and gpu_id < NUM_GPUS:
					future = executor.submit(process_images_batch, gpu_id, gpu_batch)
					future_to_batch[future] = (batch_idx, gpu_id, gpu_batch)
		
		# Collect results as they complete
		for future in as_completed(future_to_batch):
			batch_idx, gpu_id, gpu_batch = future_to_batch[future]
			try:
				batch_results = future.result()
				all_results.extend(batch_results)
				completed_images += len(batch_results)
				
				elapsed = time.time() - start_time
				rate = completed_images / elapsed if elapsed > 0 else 0
				print(f"GPU {gpu_id}: Batch {batch_idx} done. Progress: {completed_images}/{len(files)} ({rate:.1f} img/s)")
				
			except Exception as e:
				print(f"GPU {gpu_id}: Batch {batch_idx} failed: {e}")
	
	# Save all results
	print(f"💾 Saving results for {split_name} split...")
	for img_path, lines, colors in all_results:
		# Copy image
		basename = os.path.basename(img_path)
		out_img_path = os.path.join(im_out, basename)
		if not os.path.exists(out_img_path):
			safe_copy(img_path, out_img_path)
		
		# Write labels
		stem = os.path.splitext(basename)[0]
		lbl_path = os.path.join(lbl_out, f"{stem}.txt")
		col_path = os.path.join(lbl_out, f"{stem}.colors.json")
		
		with open(lbl_path, "w", encoding="utf-8") as f:
			f.write("\n".join(lines))
		with open(col_path, "w", encoding="utf-8") as f:
			json.dump({"colors": colors}, f)
	
	total_time = time.time() - start_time
	avg_rate = len(files) / total_time if total_time > 0 else 0
	print(f"✅ {split_name} split complete: {len(files)} images in {total_time:.1f}s ({avg_rate:.1f} img/s)")


def main():
	"""Main function with multi-GPU parallel processing"""
	print("🚀 Starting Multi-GPU Auto-Labeling")
	print(f"💾 Dataset directory: {DATASET_DIR}")
	print(f"🖼️  Images directory: {IMAGES_DIR}")
	print(f"🔧 Available GPUs: {NUM_GPUS}")
	
	# Use optimal batch size based on GPU memory
	optimal_batch_size = get_optimal_batch_size()
	global BATCH_SIZE_PER_GPU
	BATCH_SIZE_PER_GPU = optimal_batch_size
	
	print(f"⚙️  Optimal batch size per GPU: {BATCH_SIZE_PER_GPU}")
	print(f"👥 Max workers: {MAX_WORKERS}")
	
	if torch.cuda.is_available():
		for i in range(NUM_GPUS):
			gpu_name = torch.cuda.get_device_name(i)
			gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
			print(f"🎮 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
	
	start_time = time.time()
	ensure_dirs()
	
	# Get all image files
	image_files = sorted(glob(os.path.join(IMAGES_DIR, "*.png")))
	if not image_files:
		print(f"❌ No PNG images found in {IMAGES_DIR}")
		return
	
	print(f"📊 Found {len(image_files)} images to process")
	
	# Split into train/val
	train_files, val_files = split_train_val(image_files)
	print(f"📂 Train: {len(train_files)} images, Val: {len(val_files)} images")
	
	# Process each split in parallel
	splits = [
		("train", train_files, IM_TRAIN, LBL_TRAIN),
		("val", val_files, IM_VAL, LBL_VAL),
	]
	
	for split_name, files, im_out, lbl_out in splits:
		if files:
			process_split_parallel(split_name, files, im_out, lbl_out)
	
	total_time = time.time() - start_time
	total_images = len(image_files)
	avg_rate = total_images / total_time if total_time > 0 else 0
	
	print(f"\n🎉 Auto-labeling complete!")
	print(f"📊 Total: {total_images} images in {total_time:.1f}s ({avg_rate:.1f} img/s)")
	print(f"💾 Dataset saved to: {DATASET_DIR}")
	print(f"🚀 Speedup achieved with {NUM_GPUS} GPUs!")


if __name__ == "__main__":
	# Set multiprocessing start method for CUDA compatibility
	mp.set_start_method('spawn', force=True)
	main()
