"""Detect one ArUco marker and return its offset from the camera in metres."""
import sys

import cv2
import numpy as np
import yaml


def load_camera(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    return np.array(c["camera_matrix"], dtype=np.float64), np.array(c["dist_coeffs"], dtype=np.float64)


def marker_offset(frame, K, dist, marker_id, marker_size_m, dictionary=cv2.aruco.DICT_4X4_50):
    """Return (x, y, z) of the marker centre in the camera frame, or None if not seen."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary),
                                       cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    hits = [c for c, i in zip(corners, ids.flatten()) if i == marker_id]
    if not hits:
        return None
    s = marker_size_m / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    ok, _, tvec = cv2.solvePnP(obj, hits[0].reshape(4, 2), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    x, y, z = tvec.flatten()
    return float(x), float(y), float(z)


if __name__ == "__main__":
    # Self-check on a saved frame: python aruco_pose.py frame.jpg camera.yaml [marker_id] [size_m]
    frame = cv2.imread(sys.argv[1])
    K, dist = load_camera(sys.argv[2])
    mid = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    size = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
    print(marker_offset(frame, K, dist, mid, size))
