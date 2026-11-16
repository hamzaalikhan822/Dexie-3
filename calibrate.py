"""One-off camera calibration -> camera.yaml (DEXI 3 rosbridge camera by default).

python calibrate.py --nominal                               # Pi Camera 3 at 640x480, no capture
python calibrate.py --tag                                   # refine fx using the AprilTag pad
python calibrate.py --cols 9 --rows 6 --square 0.022         # full calibration with a checkerboard
Captures automatically every --interval seconds while the target is detected and has moved,
and finishes at --captures (SPACE and q still work as manual capture / finish).
"""
import argparse
import time

import cv2
import numpy as np
import yaml

from mission import LatestFrame


def calibrate(objpoints, imgpoints, size):
    err, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    return {"width": size[0], "height": size[1], "camera_matrix": K.tolist(),
            "dist_coeffs": dist.reshape(1, -1).tolist(), "reprojection_error": float(err)}


def tag_object_points(size_m):
    """Tag corners in detectMarkers order (TL, TR, BR, BL), z = 0."""
    return np.array([[0, 0, 0], [size_m, 0, 0], [size_m, size_m, 0], [0, size_m, 0]], np.float32)


NOMINAL_FX = 878.0  # measured on DEXI 3: 14 cm tag = 260.8 px at 0.47 m and 173.4 px at 0.71 m (640x480 stream)


def nominal(size, fx=NOMINAL_FX):
    """Pinhole at the image centre, no distortion. Good enough for a closed-loop landing."""
    K = [[fx, 0.0, size[0] / 2], [0.0, fx, size[1] / 2], [0.0, 0.0, 1.0]]
    return {"width": size[0], "height": size[1], "camera_matrix": K,
            "dist_coeffs": [[0.0] * 5], "reprojection_error": -1.0}


def calibrate_tag(objpoints, imgpoints, size):
    """One square per view cannot fix the principal point or distortion, so only fx is solved
    (centre pinned, zero distortion). A flat-only set still leaves fx unobservable, so an
    implausible answer falls back to NOMINAL_FX."""
    K0 = np.array(nominal(size)["camera_matrix"])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    err, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, K0, np.zeros(5), flags=flags)
    fx = float(K[0, 0])
    if not 450 < fx < 900:
        print(f"fx={fx:.0f} is implausible (views too flat); keeping nominal {NOMINAL_FX}")
        return nominal(size)
    result = nominal(size, fx)
    result["reprojection_error"] = float(err)
    return result


def write_camera_yaml(path, result):
    with open(path, "w") as f:
        yaml.safe_dump(result, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rtsp", default="ws://192.168.4.1:9090", help="ws://host:9090 (rosbridge) or a cv2 URL")
    p.add_argument("--topic", default="/cam0/image_raw/compressed")
    p.add_argument("--cols", type=int, default=9, help="inner corners across")
    p.add_argument("--rows", type=int, default=6, help="inner corners down")
    p.add_argument("--square", type=float, default=0.025, help="square size, metres")
    p.add_argument("--out", default="camera.yaml")
    p.add_argument("--captures", type=int, default=20, help="stop after this many")
    p.add_argument("--interval", type=float, default=1.5, help="seconds between auto captures")
    p.add_argument("--tag", action="store_true", help="calibrate with the AprilTag pad instead of a checkerboard")
    p.add_argument("--dict", default="DICT_APRILTAG_36h11")
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument("--marker-size", type=float, default=0.14)
    p.add_argument("--nominal", action="store_true", help="write the nominal 640x480 model, no capture")
    a = p.parse_args()

    if a.nominal:
        write_camera_yaml(a.out, nominal((640, 480)))
        print(f"wrote nominal {a.out}: fx={NOMINAL_FX} at (640, 480), centre, no distortion")
        return

    if a.tag:
        board = tag_object_points(a.marker_size)
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dict)),
                                           cv2.aruco.DetectorParameters())
        a.captures = max(a.captures, 30)
    else:
        board = np.zeros((a.cols * a.rows, 3), np.float32)
        board[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
    objpoints, imgpoints = [], []
    last, last_centre = 0.0, None
    if a.rtsp.startswith("dji://"):  # Mini 3 through the phone relay
        from dji_link import DjiLink
        frames = DjiLink(a.rtsp[len("dji://"):])
    else:
        frames = LatestFrame(a.rtsp, a.topic)
    frames.start()
    size = None
    while True:
        frame, _ = frames.get()
        if frame is None:
            cv2.waitKey(50)
            continue
        frame = frame.copy()
        size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if a.tag:
            cs, ids, _ = detector.detectMarkers(gray)
            hits = [c for c, i in zip(cs, ids.flatten()) if i == a.marker_id] if ids is not None else []
            found = bool(hits)
            if found:
                corners = hits[0].reshape(-1, 1, 2).astype(np.float32)
                cv2.aruco.drawDetectedMarkers(frame, [hits[0]])
        else:
            found, corners = cv2.findChessboardCorners(gray, (a.cols, a.rows))
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            if not a.tag:
                cv2.drawChessboardCorners(frame, (a.cols, a.rows), corners, found)
        cv2.putText(frame, f"captures: {len(imgpoints)}/{a.captures}  move + tilt the target", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("calibrate", frame)
        key = cv2.waitKey(1) & 0xFF
        # auto-capture only when the target has moved, so 30 identical views cannot pile up
        centre = corners.reshape(-1, 2).mean(axis=0) if found else None
        moved = found and (last_centre is None or np.linalg.norm(centre - last_centre) > 40)
        auto = found and moved and time.monotonic() - last >= a.interval
        if found and (auto or key == ord(" ")):
            objpoints.append(board)
            imgpoints.append(corners)
            last, last_centre = time.monotonic(), centre
            print(f"captured {len(imgpoints)}/{a.captures}", flush=True)
        if key == ord("q") or len(imgpoints) >= a.captures:
            break
    cv2.destroyAllWindows()
    if len(imgpoints) < 10:
        raise SystemExit(f"only {len(imgpoints)} captures, need >= 10")
    result = (calibrate_tag if a.tag else calibrate)(objpoints, imgpoints, size)
    write_camera_yaml(a.out, result)
    K = result["camera_matrix"]
    print(f"wrote {a.out}, reprojection error {result['reprojection_error']:.3f} px at {size}")
    print(f"fx={K[0][0]:.0f} fy={K[1][1]:.0f} cx={K[0][2]:.0f} cy={K[1][2]:.0f} "
          f"(centre would be {size[0] / 2:.0f}, {size[1] / 2:.0f})")


if __name__ == "__main__":
    main()
