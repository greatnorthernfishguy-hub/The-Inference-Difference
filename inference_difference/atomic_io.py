# ---- Changelog ----
# [2026-10-14] Claude Code (Opus 5.5, lane atomic-saves; Josh 2026-10-14 "We should have all saves and checkpoints,
#   anything that can't risk a failed save, be atomic") — initial helper
# What: atomic_write_bytes(path, data): temp file in the same directory -> write -> flush + fsync -> os.replace ->
#   fsync the directory.
# Why: TID's learned routing state (model_stats_cache.msgpack: success stats, penalties, TTL-exempt tool_competence)
#   and routing_state.msgpack were written with open('wb') in place; a restart / kill mid-write truncated them and
#   load_stats() then started cold ("a model proven tool-incapable must stay known").
# How: same bytes and paths; the final file keeps the existing file's permission bits.
# -------------------
"""Atomic file writes for TID's persisted state."""

from __future__ import annotations

import os


def atomic_write_bytes(path: str, data: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        mode = os.stat(path).st_mode & 0o7777
    except FileNotFoundError:
        mode = None
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
