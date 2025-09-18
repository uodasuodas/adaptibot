#!/usr/bin/env python3
"""
Check labeling progress - analyze which files have been labeled and their timestamps
"""
import os
import json
from glob import glob
from datetime import datetime
from pathlib import Path

# Paths
IMAGES_DIR = "/home/ut-ai/ai-works/adaptibot/yolo_detect/processed/images"
TRAIN_LABELS_DIR = "/home/ut-ai/ai-works/adaptibot/yolo_detect/dataset/labels/train"
VAL_LABELS_DIR = "/home/ut-ai/ai-works/adaptibot/yolo_detect/dataset/labels/val"

def get_file_info(file_path):
    """Get file modification time and size"""
    try:
        stat = os.stat(file_path)
        return {
            'mtime': stat.st_mtime,
            'mtime_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'size': stat.st_size
        }
    except OSError:
        return None

def analyze_labeling_progress():
    """Analyze which files have been labeled and when"""
    
    # Get all PNG images
    image_files = sorted(glob(os.path.join(IMAGES_DIR, "*.png")))
    image_stems = {Path(f).stem for f in image_files}
    
    print(f"Total images to label: {len(image_files)}")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Train labels directory: {TRAIN_LABELS_DIR}")
    print(f"Val labels directory: {VAL_LABELS_DIR}")
    print("-" * 80)
    
    # Get labeled files in train and val
    train_labels = sorted(glob(os.path.join(TRAIN_LABELS_DIR, "*.txt")))
    val_labels = sorted(glob(os.path.join(VAL_LABELS_DIR, "*.txt")))
    
    train_stems = {Path(f).stem for f in train_labels}
    val_stems = {Path(f).stem for f in val_labels}
    
    labeled_stems = train_stems | val_stems
    unlabeled_stems = image_stems - labeled_stems
    
    print(f"Labeled files: {len(labeled_stems)} ({len(train_labels)} train + {len(val_labels)} val)")
    print(f"Unlabeled files: {len(unlabeled_stems)}")
    print(f"Progress: {len(labeled_stems)}/{len(image_files)} ({100*len(labeled_stems)/len(image_files):.1f}%)")
    print("-" * 80)
    
    # Analyze modification times for labeled files
    labeled_files_info = []
    
    for label_file in train_labels + val_labels:
        info = get_file_info(label_file)
        if info:
            info['file'] = os.path.basename(label_file)
            info['split'] = 'train' if label_file in train_labels else 'val'
            labeled_files_info.append(info)
    
    # Sort by modification time (newest first)
    labeled_files_info.sort(key=lambda x: x['mtime'], reverse=True)
    
    # Show recently labeled files (last 50)
    print("Recently labeled files (newest first):")
    print("Time                | Split | File")
    print("-" * 60)
    for info in labeled_files_info[:50]:
        print(f"{info['mtime_str']} | {info['split']:5} | {info['file']}")
    
    if len(labeled_files_info) > 50:
        print(f"... and {len(labeled_files_info) - 50} more labeled files")
    
    print("-" * 80)
    
    # Show some unlabeled files
    if unlabeled_stems:
        print(f"Sample unlabeled files ({min(20, len(unlabeled_stems))} of {len(unlabeled_stems)}):")
        unlabeled_list = sorted(list(unlabeled_stems))
        for stem in unlabeled_list[:20]:
            print(f"  {stem}.png")
        if len(unlabeled_list) > 20:
            print(f"  ... and {len(unlabeled_list) - 20} more unlabeled files")
    
    print("-" * 80)
    
    # Find files labeled in the last hour, day, etc.
    now = datetime.now().timestamp()
    hour_ago = now - 3600
    day_ago = now - 86400
    
    recent_hour = [f for f in labeled_files_info if f['mtime'] > hour_ago]
    recent_day = [f for f in labeled_files_info if f['mtime'] > day_ago]
    
    print(f"Files labeled in the last hour: {len(recent_hour)}")
    print(f"Files labeled in the last day: {len(recent_day)}")
    
    # Save detailed report
    report = {
        'total_images': len(image_files),
        'labeled_count': len(labeled_stems),
        'unlabeled_count': len(unlabeled_stems),
        'train_count': len(train_labels),
        'val_count': len(val_labels),
        'progress_percent': 100 * len(labeled_stems) / len(image_files),
        'recent_hour_count': len(recent_hour),
        'recent_day_count': len(recent_day),
        'unlabeled_files': sorted(list(unlabeled_stems)),
        'recent_labeled': [f['file'] for f in labeled_files_info[:100]]  # Top 100 recent
    }
    
    report_file = '/home/ut-ai/ai-works/adaptibot/yolo_detect/labeling_progress_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Detailed report saved to: {report_file}")
    
    return report

if __name__ == "__main__":
    analyze_labeling_progress()
