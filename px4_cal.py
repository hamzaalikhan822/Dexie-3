"""PX4 sensor calibration over MAVSDK. The drone must be disarmed, props off for accel.

python px4_cal.py gyro     # still, any orientation, ~5 s
python px4_cal.py level    # still on a truly level surface; sets the "level" attitude
python px4_cal.py accel    # follow the prompts: hold still on each of 6 sides
DEXI 3 has no magnetometer, so there is no compass calibration.
"""
import asyncio
import os
import sys

import mission

STEPS = {"gyro": "calibrate_gyro", "level": "calibrate_level_horizon", "accel": "calibrate_accelerometer"}


async def main(kind, url):
    drone = await asyncio.wait_for(mission.connect(url), 20)
    try:
        async for a in drone.telemetry.armed():
            if a:
                print("vehicle is ARMED: refusing to calibrate")
                return
            break
        last = None
        async for p in getattr(drone.calibration, STEPS[kind])():
            line = p.status_text if p.has_status_text else f"{p.progress * 100:.0f} %"
            if line != last:
                print(line, flush=True)
                last = line
        print(f"{kind} calibration done", flush=True)
    finally:
        drone._stop_mavsdk_server()  # os._exit below would leave it holding port 50051


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        sys.exit(__doc__)
    try:
        asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "udpout://192.168.4.1:14540"))
    except Exception as ex:
        print(f"calibration failed: {ex!r}")
    sys.stdout.flush()
    os._exit(0)  # the gRPC poller thread otherwise hangs interpreter exit on Windows
