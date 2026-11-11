# Control UI — Design

Date: 2026-11-10. Builds on `2026-11-10-precision-landing-design.md`.

## Goal

A desktop window on the laptop to run the precision-landing mission and to stop it
instantly: Hold, Land, and motor cutoff (KILL).

## Decisions

- Tkinter (stdlib), one file `ui.py`. No new dependencies. Video is shown by encoding the
  frame as PPM and handing the bytes to `tk.PhotoImage`, so Pillow is not needed.
- The asyncio event loop runs in a background thread; Tk owns the main thread. Buttons
  schedule coroutines on the loop with `run_coroutine_threadsafe`.
- `mission.py` stays the single place with flight logic. The UI calls it; it does not
  copy it. Small refactors to make that possible:
  - `parse_args(argv=None)` so the UI can build `args` from its fields.
  - `connect(url)` factored out of `main()`.
  - `log = print` module function used for all mission output; the UI replaces it.
  - `Telem` also tracks `armed`.
  - `abort_action(drone, kind)` with kind in `kill | hold | land` (land keeps the retry
    and the "VEHICLE IN HOLD" message). `fly(..., ctl=None)` reads `ctl["abort"]` in its
    abort handler so a UI-triggered cancel performs the chosen action, not always land.

## Window

- Left: live camera image (scaled to 640 px wide) with a centre crosshair and, when the
  marker is seen, a circle at its projected centre and the offset text. This replaces
  `--dry`: the readout is always live once connected.
- Right, top: fields `MAVLink URL`, `RTSP URL`, `camera.yaml`, `Altitude`, `Kp`,
  `Cam yaw`, `Marker size`. `Connect` button.
- Right, middle: telemetry labels: connected, mode, armed, altitude, in air, marker offset.
- Right, bottom: `Start mission` (disabled until connected, disabled while running),
  then the emergency row: `HOLD`, `LAND`, `KILL`. KILL needs two clicks within 3 s; the
  first turns it red and relabels it `CONFIRM KILL`.
- Bottom: log text box fed by `mission.log`.

## Emergency semantics

Each emergency button does two things at once: schedules `abort_action(drone, kind)`
directly on the loop (fast path, works whether or not a mission is running), and, if a
mission task is running, sets `ctl["abort"] = kind` and cancels it so `fly()`'s handler
performs the same action (idempotent: a second stop/land/kill is harmless). KILL cuts
motors immediately; the drone falls. The pilot-in-manual and link-lost guards in
`fly()` still apply to the cancel path, never to the direct path: a human pressed it.

## Testing

- `tests/test_mission.py`: `abort_action` with a fake drone for kill, hold, land, and
  land failing twice (logs the HOLD message, returns False).
- Smoke: construct the window headless-ish (destroy after 500 ms) and render one synthetic
  frame through the PPM path.
- Hardware: same README order; the UI replaces the `--dry` step.

## Out of scope

Per-phase buttons (takeoff only, rotate only), parameter editing on the drone, a web UI.
