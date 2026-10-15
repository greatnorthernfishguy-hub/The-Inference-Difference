# ---- Changelog ----
# [2026-10-15] Claude Code (Opus 5.5, lane tid-provider-routing) — stub engine gains _provider_penalty / _pair_stats
#   (save_stats now also persists the (model, provider) pair box and per-pair window; design 2026-10-15 §3.D).
# [2026-10-14] Claude Code (Opus 5.5, lane atomic-saves; Josh 2026-10-14 "anything that can't risk a failed save, be
#   atomic") — created
#   What: SIGKILL proof for RoutingEngine.save_stats() (model_stats_cache.msgpack) and app._save_routing_state()
#         (routing_state.msgpack): a forked child runs the REAL writers in a loop; the parent SIGKILLs it at random
#         points; after every kill both files unpack with msgpack. Plus: a failing write leaves the previous file.
#   Why:  #1095 class (torn save on restart).
#   How:  stub `self` for save_stats; app's path function monkeypatched to tmp_path.
# -------------------
import os
import random
import signal
import time
import types

import msgpack
import pytest

from inference_difference import atomic_io
from inference_difference.router import RoutingEngine


def test_failed_write_keeps_previous(tmp_path, monkeypatch):
    p = tmp_path / "s.msgpack"
    atomic_io.atomic_write_bytes(str(p), b"good")
    monkeypatch.setattr(atomic_io.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk gone")))
    with pytest.raises(OSError):
        atomic_io.atomic_write_bytes(str(p), b"bad")
    assert p.read_bytes() == b"good"
    assert [n for n in os.listdir(tmp_path) if ".tmp" in n] == []


def test_sigkill_stats_and_routing_state_never_tear(tmp_path, monkeypatch):
    from inference_difference import app
    monkeypatch.setattr(app, "_routing_state_path", lambda: str(tmp_path / "routing_state.msgpack"))
    stats = {f"model-{i}": {"ok": i, "fail": 1, "pad": "x" * 200} for i in range(3000)}
    stub = types.SimpleNamespace(_stats_dirty=True, _stats_cache_path=str(tmp_path / "model_stats_cache.msgpack"),
                                 _model_success_stats=stats, _tool_competence={"m": 0.5},
                                 _model_penalty=types.SimpleNamespace(to_dict=lambda: {"p": 1}),
                                 # [2026-10-15] provider-level routing: save_stats also writes the pair box + window
                                 _provider_penalty=types.SimpleNamespace(to_dict=lambda: {}),
                                 _pair_stats=types.SimpleNamespace(to_dict=lambda: {}))
    RoutingEngine.save_stats(stub)
    app._save_routing_state()
    rng = random.Random(1095)
    bad = []
    for r in range(25):
        rfd, wfd = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(rfd)
                os.write(wfd, b"r")
                while True:
                    stub._stats_dirty = True
                    RoutingEngine.save_stats(stub)
                    app._save_routing_state()
            finally:
                os._exit(3)
        os.close(wfd)
        os.read(rfd, 1)
        os.close(rfd)
        time.sleep(rng.uniform(0.01, 0.3))
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        try:
            d = msgpack.unpackb(open(stub._stats_cache_path, "rb").read(), raw=False, strict_map_key=False)
            assert len(d["stats"]) == 3000
            msgpack.unpackb(open(tmp_path / "routing_state.msgpack", "rb").read(), raw=False)
        except Exception as exc:  # noqa: BLE001
            bad.append((r, repr(exc)))
        for n in os.listdir(tmp_path):
            if ".tmp" in n:
                os.remove(tmp_path / n)
    assert bad == []
