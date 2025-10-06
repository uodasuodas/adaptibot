import base64
import io
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from ultralytics import YOLO
from color_utils import compute_dominant_color_id, COLOR_ID_TO_NAME
import uvicorn

app = FastAPI(title="Object Detection API", description="YOLO-based object detection service")

# Load model at startup
model = None
class_names = []

def load_model():
    global model, class_names
    weights_path = "/app/weights/best.pt"
    import torch
    # Disable weights_only for compatibility with older model files
    import os
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "False"
    
    # Try loading with weights_only=False directly
    original_load = torch.load
    def compatible_load(*args, **kwargs):
        kwargs.pop('weights_only', None)  # Remove weights_only if present
        return original_load(*args, weights_only=False, **kwargs)
    torch.load = compatible_load
    
    model = YOLO(weights_path)
    
    # Restore original torch.load
    torch.load = original_load
    
    names_map = model.names
    if isinstance(names_map, dict):
        class_names = [names_map[i] for i in range(len(names_map))]
    else:
        class_names = list(names_map)

@app.on_event("startup")
async def startup_event():
    load_model()

class DetectionRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Detection confidence threshold")

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    color_name: str
    color_id: int

class DetectionResponse(BaseModel):
    detections: List[Detection]
    annotated_image: str  # Base64 encoded image with bboxes
    processing_time_ms: float

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest):
    """Detect objects in the provided image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Decode image
    image = decode_base64_image(request.image)
    
    # Run detection
    import time
    start_time = time.time()
    
    results = model.predict(image, imgsz=640, conf=request.confidence_threshold)[0]
    
    processing_time = (time.time() - start_time) * 1000
    
    # Process detections
    detections = []
    annotated_image = image.copy()
    
    for box in results.boxes:
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Compute dominant color
        color_id = compute_dominant_color_id(image, (x1, y1, x2, y2))
        color_name = COLOR_ID_TO_NAME.get(color_id, 'unknown')
        
        # Get class name
        class_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        
        # Create detection object
        detection = Detection(
            class_name=class_name,
            confidence=confidence,
            bbox=[x1, y1, x2, y2],
            color_name=color_name,
            color_id=color_id
        )
        detections.append(detection)
        
        # Draw bounding box and label
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f} | {color_name}"
        cv2.putText(annotated_image, label, (x1, max(0, y1-5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Encode annotated image
    annotated_base64 = encode_image_to_base64(annotated_image)
    
    return DetectionResponse(
        detections=detections,
        annotated_image=annotated_base64,
        processing_time_ms=processing_time
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
