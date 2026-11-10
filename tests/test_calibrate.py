import cv2
import numpy as np

from aruco_pose import load_camera
from calibrate import calibrate, write_camera_yaml


def test_calibrate_recovers_known_intrinsics(tmp_path):
    # Project a 9x6 board through a known camera from several poses, calibrate, compare.
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])
    dist = np.zeros(5)
    board = np.zeros((9 * 6, 3), np.float32)
    board[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
    objpoints, imgpoints = [], []
    rng = np.random.default_rng(0)
    for i in range(12):
        rvec = rng.normal(0, 0.3, 3)
        tvec = np.array([-0.1, -0.07, 0.6]) + rng.normal(0, 0.05, 3)
        pts, _ = cv2.projectPoints(board, rvec, tvec, K, dist)
        objpoints.append(board)
        imgpoints.append(pts.astype(np.float32))
    result = calibrate(objpoints, imgpoints, (640, 480))
    out = tmp_path / "camera.yaml"
    write_camera_yaml(str(out), result)
    K2, dist2 = load_camera(str(out))
    assert np.allclose(K2, K, atol=5)
    assert result["reprojection_error"] < 0.5
