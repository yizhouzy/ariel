"""color tester
"""

import cv2, numpy as np
pixel = np.array([[[255, 102, 0]]], dtype=np.uint8)
print(cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV))  # → H≈12, S=255, V=255