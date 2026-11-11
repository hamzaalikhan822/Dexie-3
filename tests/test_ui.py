import time

import numpy as np
import pytest

from test_aruco_pose import DIST, K, synthetic_frame

tk = pytest.importorskip("tkinter")


def test_window_renders_frame_and_marker_readout():
    import ui
    try:
        app = ui.App()
    except tk.TclError as ex:  # no display
        pytest.skip(str(ex))
    app.K, app.dist = K, DIST
    frame = synthetic_frame(cx=420)
    app.frames = type("F", (), {"get": staticmethod(lambda: (frame, time.monotonic()))})()
    app.log("hello")
    app.tick()
    text = app.status.cget("text")
    assert "marker x=+" in text and "cmd    fwd=" in text
    assert app.photo is not None and app.photo.width() == ui.VIEW_W
    assert "hello" in app.logbox.get("1.0", "end")
    assert str(app.start_btn.cget("state")) == "disabled"
    app.root.destroy()


def test_kill_needs_two_clicks():
    import ui
    try:
        app = ui.App()
    except tk.TclError as ex:
        pytest.skip(str(ex))
    sent = []
    app.drone = object()
    app.submit = lambda coro: (sent.append(coro), coro.close())
    app.kill()
    assert sent == [] and "CONFIRM" in app.kill_btn.cget("text")
    app.kill()
    assert len(sent) == 1 and app.ctl["abort"] == "kill"
    app.root.destroy()
