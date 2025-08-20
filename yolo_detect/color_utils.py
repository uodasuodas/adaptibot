import cv2
import numpy as np
from typing import Tuple, Dict

# Color IDs
# 0 any/unknown
# 1 red, 2 blue, 3 green, 4 yellow, 5 black, 6 grey, 7 white, 8 violet, 9 orange, 10 turquoise
COLOR_ID_TO_NAME: Dict[int, str] = {
	0: "any/unknown",
	1: "red",
	2: "blue",
	3: "green",
	4: "yellow",
	5: "black",
	6: "grey",
	7: "white",
	8: "violet",
	9: "orange",
	10: "turquoise",
}

# Reference HSV (OpenCV ranges: H: [0, 180], S: [0,255], V: [0,255])
# These are coarse centers; we pick nearest center by weighted distance
REFERENCE_HSV = {
	1: (0, 200, 150),     # red (wrap around 0/180)
	2: (120, 180, 150),   # blue
	3: (60, 180, 150),    # green
	4: (30, 200, 180),    # yellow
	5: (0, 0, 20),        # black
	6: (0, 10, 127),      # grey
	7: (0, 5, 240),       # white
	8: (150, 150, 150),   # violet/purple
	9: (20, 200, 180),    # orange
	10: (90, 180, 150),   # turquoise/cyan
}


def _hue_distance(h1: float, h2: float) -> float:
	# circular distance on hue ring [0,180]
	diff = abs(h1 - h2)
	return min(diff, 180 - diff)


def map_hsv_to_color_id(hsv_mean: Tuple[float, float, float]) -> int:
	"""
	Map mean HSV to one of the predefined color IDs. Return 0 if unclear.
	"""
	h, s, v = hsv_mean
	# First, handle black/white/grey by V and S thresholds
	if v < 40:
		return 5  # black
	if s < 25 and v > 220:
		return 7  # white
	if s < 30 and 50 <= v <= 220:
		return 6  # grey
	# For chromatic colors, pick nearest reference by weighted distance
	best_id = 0
	best_score = 1e9
	for cid, (rh, rs, rv) in REFERENCE_HSV.items():
		if cid in (5, 6, 7):
			# handled above
			continue
		hue_d = _hue_distance(h, rh)
		s_d = abs(s - rs) / 255.0
		v_d = abs(v - rv) / 255.0
		# emphasize hue, then saturation, then value
		score = 2.0 * (hue_d / 180.0) + 1.0 * s_d + 0.5 * v_d
		if score < best_score:
			best_score = score
			best_id = cid
	# Heuristic: if saturation is too low, fall back to grey
	if s < 40 and best_id not in (5, 6, 7):
		return 6
	return best_id if best_id != 0 else 0


def compute_dominant_color_id(image_bgr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> int:
	"""
	Compute dominant color ID inside bbox (x1,y1,x2,y2) on BGR image.
	Returns a color ID (0..10).
	"""
	h, w = image_bgr.shape[:2]
	x1, y1, x2, y2 = bbox_xyxy
	x1 = max(0, min(int(x1), w - 1))
	y1 = max(0, min(int(y1), h - 1))
	x2 = max(0, min(int(x2), w - 1))
	y2 = max(0, min(int(y2), h - 1))
	if x2 <= x1 or y2 <= y1:
		return 0
	patch = image_bgr[y1:y2, x1:x2]
	if patch.size == 0:
		return 0
	# downsample for speed
	max_dim = 128
	ph, pw = patch.shape[:2]
	scale = min(1.0, max_dim / max(ph, pw))
	if scale < 1.0:
		patch = cv2.resize(patch, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)
	hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
	# exclude very dark pixels to avoid bias towards black shadows
	mask = hsv[:, :, 2] > 20
	if np.count_nonzero(mask) < 50:
		return 0
	mean_h = float(cv2.mean(hsv[:, :, 0], mask.astype(np.uint8))[0])
	mean_s = float(cv2.mean(hsv[:, :, 1], mask.astype(np.uint8))[0])
	mean_v = float(cv2.mean(hsv[:, :, 2], mask.astype(np.uint8))[0])
	return map_hsv_to_color_id((mean_h, mean_s, mean_v))
