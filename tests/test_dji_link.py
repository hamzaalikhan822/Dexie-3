import json
import socket
import struct

import cv2
import numpy as np
import pytest

import dji_link


def packet(state, jpeg=b""):
    h = json.dumps(state).encode()
    return struct.pack(">I", len(h)) + h + struct.pack(">I", len(jpeg)) + jpeg


def test_read_packet_decodes_state_and_jpeg():
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, :, 2] = 255  # red in BGR
    a, b = socket.socketpair()
    a.sendall(packet({"flying": True, "vs": False}, cv2.imencode(".jpg", img)[1].tobytes()))
    state, frame = dji_link.read_packet(b)
    assert state == {"flying": True, "vs": False}
    assert frame.shape == (8, 8, 3) and frame[4, 4, 2] > 200 and frame[4, 4, 0] < 50


def test_heartbeat_packet_has_no_frame():
    a, b = socket.socketpair()
    a.sendall(packet({"flying": False, "vs": False}))
    assert dji_link.read_packet(b) == ({"flying": False, "vs": False}, None)


def test_closed_socket_raises():
    a, b = socket.socketpair()
    a.close()
    with pytest.raises(ConnectionError):
        dji_link.read_packet(b)


def test_oversized_length_is_rejected():
    a, b = socket.socketpair()
    a.sendall(struct.pack(">I", 1 << 30))
    with pytest.raises(ValueError):
        dji_link.read_packet(b)


def test_send_writes_one_json_line():
    a, b = socket.socketpair()
    link = dji_link.DjiLink("unused")
    link.sock = a
    assert link.send(fwd=0.1, right=0.0, down=0.0) is True
    raw = b.recv(200)
    assert raw.endswith(b"\n") and json.loads(raw) == {"fwd": 0.1, "right": 0.0, "down": 0.0}


def test_send_without_a_link_returns_false():
    assert dji_link.DjiLink("unused").send(cmd="land") is False
