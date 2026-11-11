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


def test_single_tag_calibration_recovers_intrinsics(tmp_path):
    # 30 views of one 14 cm square (4 points each) through a known camera; the constrained
    # solver must recover fx and the principal point to within a few pixels.
    from calibrate import calibrate_tag, tag_object_points
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])  # centre pinned by the solver
    dist = np.zeros(5)
    obj = tag_object_points(0.14)
    rng = np.random.default_rng(1)
    objpoints, imgpoints = [], []
    while len(imgpoints) < 30:
        rvec = rng.uniform(-0.6, 0.6, 3)
        tvec = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2), rng.uniform(0.4, 1.2)])
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        pts = pts.reshape(-1, 2)
        if pts.min() < 0 or pts[:, 0].max() > 640 or pts[:, 1].max() > 480:
            continue
        pts += rng.normal(0, 0.3, pts.shape)  # corner noise
        objpoints.append(obj)
        imgpoints.append(pts.astype(np.float32).reshape(-1, 1, 2))
    result = calibrate_tag(objpoints, imgpoints, (640, 480))
    K2 = np.array(result["camera_matrix"])
    assert abs(K2[0, 0] - 600) < 10 and K2[1, 1] == K2[0, 0]
    assert K2[0, 2] == 320 and K2[1, 2] == 240
    assert result["reprojection_error"] < 1.0


def test_tag_calibration_falls_back_to_nominal_when_flat():
    # all views fronto-parallel: fx and distance are inseparable, so the solver must not run away
    from calibrate import NOMINAL_FX, calibrate_tag, tag_object_points
    obj = tag_object_points(0.14)
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])
    objpoints, imgpoints = [], []
    rng = np.random.default_rng(2)
    for _ in range(30):
        tvec = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.15, 0.15), rng.uniform(0.4, 1.0)])
        pts, _ = cv2.projectPoints(obj, np.zeros(3), tvec, K, np.zeros(5))
        objpoints.append(obj)
        imgpoints.append((pts.reshape(-1, 2) + rng.normal(0, 0.5, (4, 2))).astype(np.float32).reshape(-1, 1, 2))
    fx = np.array(calibrate_tag(objpoints, imgpoints, (640, 480))["camera_matrix"])[0, 0]
    assert 450 < fx < 900  # either a sane solve or the nominal fallback, never 19620


def test_nominal_model_loads(tmp_path):
    from calibrate import NOMINAL_FX, nominal, write_camera_yaml
    out = tmp_path / "camera.yaml"
    write_camera_yaml(str(out), nominal((640, 480)))
    K, dist = load_camera(str(out))
    assert K[0, 0] == NOMINAL_FX and K[0, 2] == 320 and dist.shape[-1] == 5
