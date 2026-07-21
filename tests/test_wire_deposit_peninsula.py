"""
Wire → Commons Peninsula (#80) — Task 3 (TIDPeninsulaBody.deposit_wire) + Task 4
(wire_deposit._write flip) acceptance tests.

# ---- Changelog ----
# [2026-07-21] Claude Code (Sonnet 5) — #80 wire → Commons: Task 3/4 acceptance tests
# What: Proves (1) TIDPeninsulaBody.deposit_wire sends a well-formed "wire_experience"
#       msgpack frame when connected and is a silent no-op when not; (2) wire_deposit._write
#       (and therefore deposit_outbound/deposit_inbound) routes through
#       tid_peninsula_body.get_peninsula().deposit_wire with a content-derived target_id
#       (wire:{source}:{sha256(content)[:16]}) instead of the retired ng_tract write, and
#       never raises when the peninsula is absent/disconnected.
# Why:  Design v4 / implementation plan Tasks 3 and 4. Task 3 gives TID a body-side send
#       method; Task 4 is the actual flip away from the old ng_tract-based tract write that
#       fed NeuroGraph's wire_absorption.py drain path into Syl's cognitive _memory graph as
#       per-call orphan nodes (60,142 observed — the motivating churn problem, #80).
# How:  Task 3 tests exercise TIDPeninsulaBody directly with a fake socket-like connection
#       object (records sent frames) standing in for the real Unix socket. Task 4 tests
#       monkeypatch inference_difference.tid_peninsula_body.get_peninsula (the exact import
#       _write performs) with a fake peninsula recording deposit_wire calls, so no real
#       socket or NeuroGraph process is required.
"""

import hashlib
import sys
from unittest.mock import MagicMock

import msgpack
import pytest

sys.path.insert(0, ".")

from inference_difference.tid_peninsula_body import TIDPeninsulaBody
from inference_difference import wire_deposit


# ---------------------------------------------------------------------------
# Task 3 — TIDPeninsulaBody.deposit_wire
# ---------------------------------------------------------------------------

class _FakeConn:
    """Stand-in for the Unix socket connection — records what _send_frame would send."""
    def __init__(self):
        self.sent = []

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)


def _unpack_last_frame(conn: _FakeConn):
    # _send_frame writes a 4-byte length-prefixed header + msgpack payload in one sendall.
    assert conn.sent, "nothing was sent"
    raw = conn.sent[-1]
    payload = raw[4:]  # strip the 4-byte length header
    return msgpack.unpackb(payload, raw=False)


def test_deposit_wire_sends_wire_experience_frame():
    body = TIDPeninsulaBody()
    conn = _FakeConn()
    body._conn = conn

    body.deposit_wire("GET / HTTP/1.1\r\n\r\n", "wire:tid.http.outbound:abc123", {"count": 1})

    msg = _unpack_last_frame(conn)
    assert msg["type"] == "wire_experience"
    assert msg["content"] == "GET / HTTP/1.1\r\n\r\n"
    assert msg["target_id"] == "wire:tid.http.outbound:abc123"
    assert msg["metadata"] == {"count": 1}


def test_deposit_wire_defaults_metadata_to_empty_dict():
    body = TIDPeninsulaBody()
    conn = _FakeConn()
    body._conn = conn

    body.deposit_wire("content", "wire:tid.http.inbound:def456")

    msg = _unpack_last_frame(conn)
    assert msg["metadata"] == {}


def test_deposit_wire_noop_when_not_connected():
    body = TIDPeninsulaBody()
    assert body._conn is None
    # Must not raise, must not touch _send_lock's absence of a connection.
    body.deposit_wire("content", "wire:tid.http.outbound:xyz")


def test_deposit_wire_noop_on_empty_content_or_target_id():
    body = TIDPeninsulaBody()
    conn = _FakeConn()
    body._conn = conn

    body.deposit_wire("", "wire:tid.http.outbound:xyz")
    body.deposit_wire("content", "")
    assert conn.sent == [], "empty content/target_id must never send a frame"


def test_deposit_wire_never_raises_on_send_failure():
    body = TIDPeninsulaBody()

    class _BrokenConn:
        def sendall(self, data):
            raise OSError("broken pipe")

    body._conn = _BrokenConn()
    # Must not raise -- fail-soft.
    body.deposit_wire("content", "wire:tid.http.outbound:xyz")


# ---------------------------------------------------------------------------
# Task 4 — wire_deposit._write flip to the peninsula
# ---------------------------------------------------------------------------

def test_write_routes_through_peninsula_with_content_derived_target_id(monkeypatch):
    fake_peninsula = MagicMock()
    monkeypatch.setattr(
        "inference_difference.tid_peninsula_body.get_peninsula",
        lambda: fake_peninsula,
    )

    content = "GET /v1/chat/completions HTTP/1.1\r\nhost: example.com\r\n\r\n"
    wire_deposit._write(content, source="tid.http.outbound")

    fake_peninsula.deposit_wire.assert_called_once()
    sent_content, sent_target_id = fake_peninsula.deposit_wire.call_args[0][:2]
    assert sent_content == content
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    assert sent_target_id == f"wire:tid.http.outbound:{expected_hash}"


def test_write_target_id_matches_commons_eviction_namespace_shape(monkeypatch):
    """target_id must split into exactly 3 ':'-parts (wire, tid.http.{dir}, hash) --
    the shape commons.py._evict_old_wire (#80 Task 1, NeuroGraph side) groups on via
    the first 2 parts. A mismatch here would silently break wire retention windowing."""
    fake_peninsula = MagicMock()
    monkeypatch.setattr(
        "inference_difference.tid_peninsula_body.get_peninsula",
        lambda: fake_peninsula,
    )

    wire_deposit._write("HTTP/1.1 200 OK\r\n\r\nok", source="tid.http.inbound")

    _, sent_target_id = fake_peninsula.deposit_wire.call_args[0][:2]
    parts = sent_target_id.split(":")
    assert len(parts) == 3
    assert parts[0] == "wire"
    assert parts[1] == "tid.http.inbound"


def test_write_is_noop_when_peninsula_not_started(monkeypatch):
    monkeypatch.setattr(
        "inference_difference.tid_peninsula_body.get_peninsula",
        lambda: None,
    )
    # Must not raise when the peninsula singleton hasn't been created yet.
    wire_deposit._write("content", source="tid.http.outbound")


def test_write_is_noop_on_empty_content(monkeypatch):
    fake_peninsula = MagicMock()
    monkeypatch.setattr(
        "inference_difference.tid_peninsula_body.get_peninsula",
        lambda: fake_peninsula,
    )
    wire_deposit._write("", source="tid.http.outbound")
    fake_peninsula.deposit_wire.assert_not_called()


def test_write_never_raises_when_peninsula_call_errors(monkeypatch):
    fake_peninsula = MagicMock()
    fake_peninsula.deposit_wire.side_effect = RuntimeError("boom")
    monkeypatch.setattr(
        "inference_difference.tid_peninsula_body.get_peninsula",
        lambda: fake_peninsula,
    )
    # Must not raise -- fail-soft, no fallback to the old tract write.
    wire_deposit._write("content", source="tid.http.outbound")


def test_deposit_outbound_builds_request_text_and_writes(monkeypatch):
    captured = {}

    def _fake_write(content, source):
        captured["content"] = content
        captured["source"] = source

    monkeypatch.setattr(wire_deposit, "_write", _fake_write)

    wire_deposit.deposit_outbound(
        provider="anthropic",
        model_id="claude-x",
        url="https://api.example.com/v1/messages",
        method="post",
        request_body={"hello": "world"},
        headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
    )

    assert captured["source"] == "tid.http.outbound"
    assert captured["content"].startswith("POST https://api.example.com/v1/messages HTTP/1.1")
    assert "Authorization: <scrubbed>" in captured["content"]
    assert "secret" not in captured["content"]
    assert '"hello": "world"' in captured["content"]


def test_deposit_inbound_builds_response_text_and_writes(monkeypatch):
    captured = {}

    def _fake_write(content, source):
        captured["content"] = content
        captured["source"] = source

    monkeypatch.setattr(wire_deposit, "_write", _fake_write)

    wire_deposit.deposit_inbound(
        provider="anthropic",
        model_id="claude-x",
        url="https://api.example.com/v1/messages",
        status_code=200,
        response_body="ok",
        reason="OK",
    )

    assert captured["source"] == "tid.http.inbound"
    assert captured["content"].startswith("HTTP/1.1 200 OK")


if __name__ == "__main__":
    test_deposit_wire_sends_wire_experience_frame();                         print("PASS deposit_wire sends wire_experience frame")
    test_deposit_wire_defaults_metadata_to_empty_dict();                     print("PASS deposit_wire defaults metadata to {}")
    test_deposit_wire_noop_when_not_connected();                             print("PASS deposit_wire no-op when not connected")
    test_deposit_wire_noop_on_empty_content_or_target_id();                  print("PASS deposit_wire no-op on empty content/target_id")
    test_deposit_wire_never_raises_on_send_failure();                       print("PASS deposit_wire never raises on send failure")
    print("\nTask 3 (TIDPeninsulaBody.deposit_wire): ALL PASS")
    print("\n(Task 4 tests use pytest monkeypatch fixtures -- run via `pytest` for those.)")
