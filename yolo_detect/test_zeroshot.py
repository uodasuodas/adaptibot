#!/usr/bin/env python3
"""
Test script to debug zero-shot detection performance.
Shows what the detector finds for each candidate label.
"""

import os
import cv2
import torch
from transformers import pipeline
from glob import glob

# Test with current labels
ZS_LABELS = [
    "can", "soda can", "tin can",
    "rubber duck", "duck",
    "cup", "mug", "plastic cup", "paper cup",
    "sponge", "kitchen sponge", "dish sponge",
    "ball", "tennis ball",
    "vegetable", "carrot", "tomato", "cucumber", "pepper",
]

def test_detection():
    device = 0 if torch.cuda.is_available() else -1
    detector = pipeline(
        "zero-shot-object-detection",
        model="google/owlvit-base-patch16",
        device=device,
    )
    
    # Get a sample image
    processed_dir = "/home/ut-ai/ai-works/adaptibot/yolo_detect/processed/images"
    images = glob(os.path.join(processed_dir, "*.png"))
    
    if not images:
        print("No images found in processed/images/")
        return
        
    # Test on first few images
    for img_path in images[:3]:
        print(f"\n=== Testing {os.path.basename(img_path)} ===")
        
        results = detector(img_path, candidate_labels=ZS_LABELS)
        
        print(f"Found {len(results)} detections:")
        for r in results:
            label = r["label"]
            score = r["score"]
            print(f"  {label}: {score:.3f}")
            
        # Show detections above different thresholds
        for thresh in [0.1, 0.25, 0.4]:
            filtered = [r for r in results if r["score"] >= thresh]
            print(f"Above {thresh}: {len(filtered)} detections")
            for r in filtered:
                print(f"  {r['label']}: {r['score']:.3f}")

if __name__ == "__main__":
    test_detection()
