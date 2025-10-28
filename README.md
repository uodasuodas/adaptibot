# AdaptiBot

AI-powered robot control system with object detection and natural language interface.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Robot Chat App](#robot-chat-app)
4. [Detection API](#detection-api)
5. [Training Pipeline](#training-pipeline)
6. [Project Structure](#project-structure)

## Overview

**Two main components:**

### 1. Robot Chat App
Natural language control interface for robot arm manipulation:
- Chat with GPT-4 to control robot movements
- Real-time object detection and tracking
- Dynamic drop point configuration
- WebSocket-based reactive UI

### 2. Object Detection Pipeline
Complete YOLO training and inference system:
- Zero-shot auto-labeling with GroundingDINO
- Color-aware object detection (9 colors)
- Containerized REST API
- Support for: can, duck, cup, sponge, ball, vegetable

## Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key (for robot chat)
- Python 3.8+ (for training)

### Robot Chat App
```bash
cd robot_chat_app
cp env.example .env
# Edit .env and add your OPENAI_API_KEY
docker-compose up --build -d
```
Access at: http://localhost:8082

### Detection API
```bash
cd detection_api
docker-compose up --build -d
```
Access at: http://localhost:8000

## Robot Chat App

Natural language interface for robot arm control via TCP/WebSocket.

### Architecture
- **Backend**: FastAPI + LangGraph (TCP server on port 9000)
- **Frontend**: Real-time WebSocket UI
- **Robot**: TCP client (tasker) connects to backend

### Protocol
```
Backend → Robot: {"cmd":"give_objects"}
Robot → Backend: {"objects": ["id color x y z", ...]}
Backend → Robot: {"cmd_list":["GRAB x y z", "DROP x y z", ...]}
Robot → Backend: {"cmd":"start_listen"}
```

### Features
- **Natural language commands**: "Put red can in BOX1"
- **GPT-4 agent**: Interprets requests and generates robot commands
- **Real-time UI**: Shows detected objects and box contents
- **Voice input**: Speak commands via browser
- **Dynamic boxes**: Robot can configure drop points

### Object Format
Objects: `"id color x y z"` where:
- `id`: 0-6 (unknown, can, duck, cup, sponge, ball, vegetable)
- `color`: 0-8 (black, white, grey, red, blue, green, cyan, magenta, yellow)
- `x,y,z`: Integer coordinates

### Default Drop Boxes
- BOX1: `50 60 14`
- BOX2: `550 160 15`
- BOX3: `850 180 16`

### Configuration
Environment variables in `.env`:
```bash
OPENAI_API_KEY=sk-...
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8082
ROBOT_SERVER_PORT=9000
```

### Documentation
- `robot_chat_app/frontend/` - WebSocket client code
- `robot_chat_app/backend/` - Server and LLM agent

## Detection API

FastAPI service for YOLO-based object detection with color analysis.

### Endpoints

**POST /detect** - Detect objects in base64 image
```json
{
  "image": "base64_encoded_image",
  "confidence_threshold": 0.25
}
```

**GET /health** - Health check

### Usage
```python
import requests, base64

with open("image.jpg", "rb") as f:
    img = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/detect",
    json={"image": img, "confidence_threshold": 0.3})
```

### Interactive Demo
```bash
cd detection_api
jupyter notebook api_demo.ipynb
```

## Training Pipeline

### Auto-Labeling
Zero-shot labeling with GroundingDINO and SAM:

```bash
cd yolo_detect
pip install -r requirements.txt

python auto_label_zeroshot.py \
  --input_folder dataset/images/unlabeled \
  --output_folder dataset/labels/train \
  --confidence_threshold 0.35
```

### YOLO Training
```bash
python train_yolo.py
```

**Key parameters:**
- Epochs: 100-300
- Image size: 640x640
- Batch size: 8-16 (adjust for GPU memory)
- Models: yolov8n (fast), yolov8s (balanced), yolov8m (accurate)

**Outputs:** `runs/stereo_objects5/weights/best.pt`

### Evaluation
```bash
cd yolo_detect
jupyter notebook eval_demo.ipynb
```

## Project Structure

```
adaptibot/
├── robot_chat_app/             # AI robot control interface
│   ├── backend/
│   │   ├── app.py              # FastAPI + WebSocket server
│   │   ├── robot_server.py     # TCP server (port 9000)
│   │   ├── llm_agent.py        # LangGraph GPT-4 agent
│   │   └── requirements.txt
│   ├── frontend/
│   │   ├── index.html          # WebSocket UI
│   │   ├── script.js
│   │   └── style.css
│   ├── docker-compose.yml
│   └── PROTOCOL.md             # TCP protocol spec
├── detection_api/              # YOLO inference API
│   ├── app.py                  # FastAPI service
│   ├── color_utils.py          # Color detection
│   ├── Dockerfile
│   └── docker-compose.yml
├── yolo_detect/                # Training pipeline
│   ├── auto_label_zeroshot.py  # Auto-labeling
│   ├── train_yolo.py           # YOLO training
│   ├── eval_demo.ipynb         # Evaluation
│   ├── dataset/                # Training data
│   └── runs/                   # Model outputs
└── README.md
```

## Troubleshooting

**Docker issues:**
```bash
docker-compose logs              # Check logs
docker-compose ps                # Check status
docker system prune -a           # Clear cache
docker-compose build --no-cache  # Fresh build
```

**Robot chat:**
- Verify `OPENAI_API_KEY` in `.env`
- Check robot connects to port 9000
- Review `PROTOCOL.md` for message format

**Detection API:**
- Ensure model weights exist: `yolo_detect/runs/stereo_objects5/weights/best.pt`
- Test with: `curl http://localhost:8000/health`

**Training:**
- Reduce batch size if CUDA out of memory
- Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Monitor: `nvidia-smi -l 1`