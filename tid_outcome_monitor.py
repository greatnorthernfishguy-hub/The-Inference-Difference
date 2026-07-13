#!/usr/bin/env python3
"""TID outcome monitor (read-only). Tails inference_difference.log, classifies
every model-call outcome by (provider, error_class, model), and writes a rolling
summary so provider/model reliability is a glance, not a forensic dig.

Read-only: never touches TID's process, config, routing, or the log (offset-tailed).
Bounded: aggregates in RAM + a capped recent-events ring. Stamps observation time
(log lines carry no timestamp); as a live tailer, obs-time ≈ event-time.

Punchlist #384 companion. Started 2026-07-11 (CC/Opus).
"""
import json, os, re, time
from collections import defaultdict, deque

LOG = os.path.expanduser("~/The-Inference-Difference/logs/inference_difference.log")
OUT_SUMMARY = os.path.expanduser("~/The-Inference-Difference/data/tid_outcome_summary.json")
OUT_EVENTS = os.path.expanduser("~/The-Inference-Difference/data/tid_outcome_events.jsonl")
OFFSET_FILE = os.path.expanduser("~/The-Inference-Difference/data/.tid_monitor_offset")
FLUSH_SECS = 60
WINDOW_SECS = 24 * 3600           # rolling window for the summary
EVENT_RING = 400                  # last N raw events kept on disk

FAIL_RE = re.compile(r"^(Model call failed|Stream failed|Stream call failed|Anthropic call failed):\s*(\S+)\s+(\d{3})\s+—\s*(.*)$")
CIRCUIT_RE = re.compile(r"^Provider circuit armed:\s*(\S+)")

def classify(model, code, body):
    """(provider, error_class) from the failure signature — the mapping proven
    2026-07-11: 'No cookie auth' is OpenRouter no-key; x402/Diem/venice is Venice."""
    b = body or ""
    if "No cookie auth credentials found" in b:
        return "openrouter", "NO_AUTH_HEADER"       # empty OPENROUTER_API_KEY at call site
    if "No auth credentials" in b:
        return "openrouter", "NO_AUTH_HEADER"
    if "x402" in b or "api.venice.ai" in b or "Insufficient USD" in b or "Diem" in b:
        return "venice", "CREDIT_402"
    if "No endpoints found" in b:
        return "unknown", "NO_ENDPOINT_404"
    if code == "401":
        return "unknown", "AUTH_401"
    if code == "402":
        return "unknown", "CREDIT_402"
    if code == "404":
        return "unknown", "NO_ENDPOINT_404"
    if code == "429":
        return "unknown", "RATE_LIMIT_429"
    return "unknown", f"OTHER_{code}"

def load_offset():
    try:
        with open(OFFSET_FILE) as f:
            off, inode = f.read().split()
            return int(off), int(inode)
    except Exception:
        return 0, 0

def save_offset(off, inode):
    tmp = OFFSET_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{off} {inode}")
    os.replace(tmp, OFFSET_FILE)

def main():
    # (provider, error_class) -> {count, models:{model:count}, first, last}
    agg = defaultdict(lambda: {"count": 0, "models": defaultdict(int), "first": None, "last": None})
    circuits = defaultdict(lambda: {"count": 0, "last": None})
    events = deque(maxlen=EVENT_RING)
    off, inode = load_offset()
    last_flush = 0.0

    def flush():
        now = time.time()
        cutoff = now - WINDOW_SECS
        # prune window
        rows = []
        for (prov, ecls), d in agg.items():
            if d["last"] and d["last"] >= cutoff:
                top = sorted(d["models"].items(), key=lambda x: -x[1])[:8]
                rows.append({"provider": prov, "error_class": ecls, "count": d["count"],
                             "first_seen": d["first"], "last_seen": d["last"],
                             "top_models": [{"model": m, "n": n} for m, n in top]})
        rows.sort(key=lambda r: -r["count"])
        summary = {
            "generated_at": int(now),
            "window_hours": WINDOW_SECS // 3600,
            "note": "read-only tail of TID log; NO_AUTH_HEADER=OpenRouter call sent with no key; CREDIT_402=Venice unfunded",
            "failures_by_class": rows,
            "circuits_armed": sorted(
                [{"provider": p, "count": d["count"], "last_seen": d["last"]}
                 for p, d in circuits.items() if d["last"] and d["last"] >= cutoff],
                key=lambda r: -r["count"]),
        }
        tmp = OUT_SUMMARY + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, OUT_SUMMARY)

    while True:
        try:
            st = os.stat(LOG)
            if st.st_ino != inode:                 # rotated/replaced
                off, inode = 0, st.st_ino
            if st.st_size < off:                   # truncated
                off = 0
            with open(LOG, "r", errors="replace") as f:
                f.seek(off)
                for line in f:
                    if not line.endswith("\n"):
                        break                      # partial tail line; wait
                    off += len(line.encode("utf-8", "replace"))
                    line = line.rstrip("\n")
                    m = FAIL_RE.match(line)
                    if m:
                        _kind, model, code, body = m.groups()
                        prov, ecls = classify(model, code, body)
                        d = agg[(prov, ecls)]
                        now = int(time.time())
                        d["count"] += 1; d["models"][model] += 1
                        d["first"] = d["first"] or now; d["last"] = now
                        events.append({"t": now, "provider": prov, "class": ecls,
                                       "model": model, "code": code})
                        continue
                    c = CIRCUIT_RE.match(line)
                    if c:
                        p = c.group(1); circuits[p]["count"] += 1
                        circuits[p]["last"] = int(time.time())
                inode = st.st_ino
            now = time.time()
            if now - last_flush >= FLUSH_SECS:
                flush()
                save_offset(off, inode)
                with open(OUT_EVENTS + ".tmp", "w") as f:
                    for e in events:
                        f.write(json.dumps(e) + "\n")
                os.replace(OUT_EVENTS + ".tmp", OUT_EVENTS)
                last_flush = now
        except FileNotFoundError:
            pass
        except Exception as exc:
            with open(os.path.expanduser("~/The-Inference-Difference/data/.tid_monitor_err"), "a") as f:
                f.write(f"{int(time.time())} {exc!r}\n")
        time.sleep(5)

if __name__ == "__main__":
    main()
