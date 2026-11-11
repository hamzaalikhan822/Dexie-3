"""One-off checkerboard calibration of the RTSP stream -> camera.yaml.

Usage: python calibrate.py --cols 9 --rows 6 --square 0.025   (DEXI 3 rosbridge camera by default)
Press SPACE to capture when corners are drawn, q to finish (needs >= 10 captures).
"""
import argparse

import cv2
import numpy as np
import yaml

from mission import LatestFrame


def calibrate(objpoints, imgpoints, size):
    err, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    return {"width": size[0], "height": size[1], "camera_matrix": K.tolist(),
            "dist_coeffs": dist.reshape(1, -1).tolist(), "reprojection_error": float(err)}


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
    a = p.parse_args()

    board = np.zeros((a.cols * a.rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
    objpoints, imgpoints = [], []
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
        found, corners = cv2.findChessboardCorners(gray, (a.cols, a.rows))
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.drawChessboardCorners(frame, (a.cols, a.rows), corners, found)
        cv2.putText(frame, f"captures: {len(imgpoints)}  SPACE=capture q=finish", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("calibrate", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            objpoints.append(board)
            imgpoints.append(corners)
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()
    if len(imgpoints) < 10:
        raise SystemExit(f"only {len(imgpoints)} captures, need >= 10")
    result = calibrate(objpoints, imgpoints, size)
    write_camera_yaml(a.out, result)
    print(f"wrote {a.out}, reprojection error {result['reprojection_error']:.3f} px at {size}")


if __name__ == "__main__":
    main()
