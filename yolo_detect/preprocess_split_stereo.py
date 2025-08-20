import os
import cv2
from glob import glob
from tqdm import tqdm

RAW_DIR = "/Users/tomas.komar/digitecus/fotos/raw"
OUT_DIR = "/Users/tomas.komar/digitecus/fotos/processed/images"


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def split_image(path: str):
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	if img is None:
		return []
	h, w = img.shape[:2]
	mid = w // 2
	left = img[:, :mid]
	right = img[:, mid:]
	return [left, right]


def main():
	ensure_dir(OUT_DIR)
	paths = sorted(glob(os.path.join(RAW_DIR, "*.png")))
	for p in tqdm(paths, desc="Splitting stereo images"):
		name = os.path.splitext(os.path.basename(p))[0]
		parts = split_image(p)
		if not parts:
			continue
		left, right = parts
		cv2.imwrite(os.path.join(OUT_DIR, f"{name}_L.png"), left)
		cv2.imwrite(os.path.join(OUT_DIR, f"{name}_R.png"), right)
	print(f"Saved split images to {OUT_DIR}")


if __name__ == "__main__":
	main()
