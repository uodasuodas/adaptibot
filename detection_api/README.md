# Object detection API

FastAPI service for YOLO-based object detection with color analysis.

## Quick start

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. API will be available at `http://localhost:8000`
3. Interactive docs at `http://localhost:8000/docs`

## API endpoints

- `POST /detect` - Detect objects in base64 image
- `GET /health` - Health check

## Request format

```json
{
  "image": "base64_encoded_image_data",
  "confidence_threshold": 0.25
}
```

## Response format

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
