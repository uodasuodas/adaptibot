# AdaptiBot Object Detection Pipeline

A complete pipeline for automated stereo camera object detection with color analysis, featuring zero-shot auto-labeling, YOLO training, and a containerized inference API.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Detection API](#detection-api)
4. [Auto Data Labeling](#auto-data-labeling)
5. [YOLO Training](#yolo-training)
6. [Project Structure](#project-structure)
7. [Troubleshooting](#troubleshooting)

## Overview

This project provides an end-to-end object detection pipeline optimized for stereo camera data:

- **Zero-shot auto-labeling** using GroundingDINO and SAM models
- **Automated YOLO training** pipeline with validation
- **Color-aware object detection** with 10 predefined color categories
- **Containerized REST API** for production inference
- **Interactive Jupyter notebooks** for training evaluation and API testing

### Supported Objects
- Cups, bottles, bowls, and other kitchen/tableware items
- 10 color categories: red, green, blue, yellow, orange, purple, pink, brown, black, white

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.8+ (for training pipeline)
- CUDA-compatible GPU (recommended for training)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd adaptibot
```

### 2. Start Detection API
```bash
cd detection_api
docker-compose up --build -d
```

### 3. Test API
```bash
# Health check
curl http://localhost:8000/health

# View interactive docs
open http://localhost:8000/docs
```

## Detection API

### Building the Docker Container

#### Build from source:
```bash
cd detection_api
docker-compose build
```

#### Start the service:
```bash
docker-compose up -d
```

#### Stop the service:
```bash
docker-compose down
```

### API Endpoints

#### Health Check
```bash
GET /health
```
Response:
```json
{"status": "healthy", "model_loaded": true}
```

#### Object Detection
```bash
POST /detect
Content-Type: application/json
```

Request body:
```json
{
  "image": "base64_encoded_image_data",
  "confidence_threshold": 0.25
}
```

Response:
```json
{
  "detections": [
    {
      "class_name": "cup",
      "confidence": 0.85,
      "bbox": [100, 150, 200, 250],
      "color_name": "red",
      "color_id": 1
    }
  ],
  "annotated_image": "base64_encoded_annotated_image",
  "processing_time_ms": 45.2
}
```

### Using the API

#### Python Example
```python
import requests
import base64

# Load and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make detection request
response = requests.post(
    "http://localhost:8000/detect",
    json={
        "image": image_data,
        "confidence_threshold": 0.3
    }
)

detections = response.json()
print(f"Found {len(detections['detections'])} objects")
```

#### cURL Example
```bash
# Encode image to base64
IMAGE_B64=$(base64 -w 0 image.jpg)

# Make detection request
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"$IMAGE_B64\",
    \"confidence_threshold\": 0.25
  }"
```

### Docker Configuration

#### Environment Variables
- `QT_QPA_PLATFORM=offscreen`: Headless OpenCV operation
- `DISPLAY=:0`: X11 display for compatibility

#### Volume Mounts
The model weights are mounted from the host:
```yaml
volumes:
  - ../yolo_detect/runs/stereo_objects5/weights:/app/weights:ro
```

#### Port Mapping
- Container port `8000` → Host port `8000`

### Interactive Demo

Run the Jupyter notebook demo:
```bash
cd detection_api
jupyter notebook api_demo.ipynb
```

Features:
- File upload widget for testing images
- Confidence threshold slider
- Real-time detection visualization
- Batch testing on validation images

## Auto Data Labeling

The auto-labeling pipeline uses zero-shot models to automatically generate YOLO format labels from stereo camera images.

### Models Used
- **GroundingDINO**: Zero-shot object detection with text prompts
- **Segment Anything (SAM)**: Precise object segmentation
- **Color Analysis**: HSV-based color classification

### Running Auto-Labeling

#### Setup Environment
```bash
cd yolo_detect
pip install -r requirements.txt
```

#### Basic Usage
```bash
python auto_label_zeroshot.py \
  --input_folder dataset/images/unlabeled \
  --output_folder dataset/labels/train \
  --confidence_threshold 0.35
```

#### Advanced Options
```bash
python auto_label_zeroshot.py \
  --input_folder dataset/images/unlabeled \
  --output_folder dataset/labels/train \
  --confidence_threshold 0.35 \
  --text_prompt "cup . bowl . bottle . mug" \
  --color_analysis \
  --visualize \
  --batch_size 8
```

### Text Prompts
The system uses customizable text prompts for object detection:

**Default prompt**: `"cup . bowl . bottle . mug . glass . plate . spoon . fork . knife . pot . pan . lid"`

**Custom prompts**:
```bash
--text_prompt "cup . bottle . bowl"  # Specific objects only
--text_prompt "red cup . blue bottle"  # Color-specific detection
```

### Output Format
Generated labels follow YOLO format:
```
# filename.txt
class_id center_x center_y width height
0 0.5 0.3 0.2 0.4
1 0.8 0.7 0.15 0.25
```

### Monitoring Progress
```bash
python check_labeling_progress.py
```

Generates:
- Progress report in JSON format
- Summary statistics
- Class distribution analysis

### Quality Control
- **Confidence filtering**: Remove low-confidence detections
- **Size filtering**: Remove tiny or oversized bounding boxes
- **Overlap removal**: NMS to eliminate duplicate detections
- **Visual inspection**: Optional visualization for manual review

## YOLO Training

### Training Pipeline

#### 1. Prepare Dataset
```bash
# Ensure proper dataset structure
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

#### 2. Configure Training
Edit `dataset.yaml`:
```yaml
train: dataset/images/train
val: dataset/images/val
nc: 10  # number of classes
names: ['cup', 'bowl', 'bottle', 'mug', 'glass', 'plate', 'spoon', 'fork', 'knife', 'other']
```

#### 3. Start Training
```bash
python train_yolo.py
```

#### Advanced Training Options
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='stereo_objects_v6',
    patience=20,
    save_period=10,
    val=True
)
```

### Training Parameters
- **Epochs**: 100-300 (depending on dataset size)
- **Image size**: 640x640 (optimal for most use cases)
- **Batch size**: 8-16 (adjust based on GPU memory)
- **Model variants**: 
  - `yolov8n.pt`: Fastest inference
  - `yolov8s.pt`: Balanced speed/accuracy
  - `yolov8m.pt`: Higher accuracy

### Model Evaluation

#### Validation Notebooks
- `eval_demo.ipynb`: Interactive model evaluation
- `label_viz.ipynb`: Dataset visualization and analysis

#### Metrics Monitoring
Training automatically generates:
- **mAP scores**: Mean Average Precision at different IoU thresholds
- **Loss curves**: Training and validation loss progression
- **Confusion matrix**: Class prediction accuracy
- **Precision/Recall curves**: Per-class performance analysis

#### Results Location
```
runs/
└── stereo_objects5/
    ├── weights/
    │   ├── best.pt      # Best model weights
    │   └── last.pt      # Final epoch weights
    ├── results.png      # Training curves
    ├── confusion_matrix.png
    └── val_batch*.jpg   # Validation predictions
```

### Hyperparameter Tuning
```python
# Custom training with hyperparameters
model.train(
    data='dataset.yaml',
    epochs=100,
    lr0=0.01,          # Initial learning rate
    momentum=0.937,     # SGD momentum
    weight_decay=0.0005,# Weight decay
    warmup_epochs=3,    # Warmup epochs
    box=7.5,           # Box loss gain
    cls=0.5,           # Class loss gain
    dfl=1.5            # DFL loss gain
)
```

## Project Structure

```
adaptibot/
├── yolo_detect/                 # Training pipeline
│   ├── auto_label_zeroshot.py   # Auto-labeling script
│   ├── train_yolo.py           # Training script
│   ├── check_labeling_progress.py
│   ├── eval_demo.ipynb         # Model evaluation
│   ├── label_viz.ipynb         # Dataset visualization
│   ├── color_utils.py          # Color detection utilities
│   ├── dataset.yaml            # Dataset configuration
│   ├── dataset/                # Training data
│   │   ├── images/
│   │   └── labels/
│   └── runs/                   # Training outputs
│       └── stereo_objects5/    # Latest model
├── detection_api/              # Production API
│   ├── app.py                  # FastAPI application
│   ├── color_utils.py          # Color detection (copy)
│   ├── Dockerfile              # Container configuration
│   ├── docker-compose.yml      # Service orchestration
│   ├── requirements.txt        # Python dependencies
│   ├── api_demo.ipynb          # Interactive API demo
│   └── README.md               # API documentation
└── README.md                   # This file
```

## Troubleshooting

### Common Issues

#### 1. Docker Build Failures
```bash
# Clear Docker cache
docker system prune -a

# Rebuild with no cache
docker-compose build --no-cache
```

#### 2. Model Loading Errors
Check model path in docker-compose.yml:
```yaml
volumes:
  - ../yolo_detect/runs/stereo_objects5/weights:/app/weights:ro
```

#### 3. CUDA Out of Memory (Training)
Reduce batch size:
```python
model.train(data='dataset.yaml', batch=4)  # Reduce from 16 to 4
```

#### 4. API Connection Issues
Verify container is running:
```bash
docker-compose ps
docker-compose logs
```

#### 5. PyTorch Compatibility
The API includes compatibility fixes for PyTorch 2.8+ with older model weights. If issues persist, try:
```bash
# Use specific PyTorch version
pip install torch==2.0.1 torchvision==0.15.2
```

### Performance Optimization

#### GPU Acceleration
For training with GPU:
```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
nvidia-smi -l 1
```

#### Memory Management
```python
# Clear GPU cache during training
import torch
torch.cuda.empty_cache()
```

#### Batch Size Tuning
Start with small batch size and increase:
```python
# Conservative: batch=4
# Moderate: batch=8  
# Aggressive: batch=16+
```

### Support

For issues and questions:
1. Check container logs: `docker-compose logs`
2. Verify model weights exist: `ls yolo_detect/runs/stereo_objects5/weights/`
3. Test with simple curl command first
4. Review Jupyter notebooks for examples

### Updates

To update the model:
1. Retrain with new data
2. Update model path in docker-compose.yml
3. Rebuild container: `docker-compose build`
4. Restart: `docker-compose up -d`