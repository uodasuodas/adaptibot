import os
import torch
from ultralytics import YOLO

DATA_YAML = "/home/ut-ai/ai-works/adaptibot/yolo_detect/dataset.yaml"
RUNS_DIR = "/home/ut-ai/ai-works/adaptibot/yolo_detect/runs"
MODEL_NAME = "yolov8n.pt"


def main():
	device = 0 if torch.cuda.is_available() else "cpu"
	model = YOLO(MODEL_NAME)
	results = model.train(
		data=DATA_YAML,
		epochs=50,
		imgsz=640,
		batch=16,
		device=device,
		project=RUNS_DIR,
		name="stereo_objects",
	)
	print(results)
	print("Best weights:", os.path.join(RUNS_DIR, "detect", "stereo_objects", "weights", "best.pt"))


if __name__ == "__main__":
	main()
