import cv2
import numpy as np

from aruco_pose import marker_offset

MARKER_PX = 200
FX = 600.0
K = np.array([[FX, 0, 320.0], [0, FX, 240.0], [0, 0, 1]], dtype=np.float64)
DIST = np.zeros((1, 5))


def synthetic_frame(cx=320, cy=240):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(d, 0, MARKER_PX)
    frame = np.full((480, 640), 255, dtype=np.uint8)
    y0, x0 = cy - MARKER_PX // 2, cx - MARKER_PX // 2
    frame[y0:y0 + MARKER_PX, x0:x0 + MARKER_PX] = marker
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def test_centered_marker_gives_zero_xy_and_expected_z():
    x, y, z = marker_offset(synthetic_frame(), K, DIST, 0, 0.35)
    # pinhole: z = f * size / pixel_size
    assert abs(z - FX * 0.35 / MARKER_PX) < 0.05
    assert abs(x) < 0.02 and abs(y) < 0.02


def test_marker_right_of_centre_gives_positive_x():
    x, y, z = marker_offset(synthetic_frame(cx=420), K, DIST, 0, 0.35)
    assert x > 0.1 and abs(y) < 0.02


def test_marker_below_centre_gives_positive_y():
    x, y, z = marker_offset(synthetic_frame(cy=340), K, DIST, 0, 0.35)
    assert y > 0.1 and abs(x) < 0.02


def test_missing_marker_returns_none():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    assert marker_offset(frame, K, DIST, 0, 0.35) is None


def test_wrong_id_returns_none():
    assert marker_offset(synthetic_frame(), K, DIST, 7, 0.35) is None


def test_apriltag_36h11_dictionary_detects_dexi_pad_pattern():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(d, 0, MARKER_PX)
    frame = np.full((480, 640), 255, dtype=np.uint8)
    frame[140:140 + MARKER_PX, 220:220 + MARKER_PX] = marker
    x, y, z = marker_offset(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), K, DIST, 0, 0.14,
                            cv2.aruco.DICT_APRILTAG_36h11)
    assert abs(z - FX * 0.14 / MARKER_PX) < 0.03
