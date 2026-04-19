import sys

NL = chr(10)
p = '/home/josh/The-Inference-Difference/inference_difference/app.py'
ls = open(p).readlines()

def find_line(ls, pattern, start=0):
    for i, l in enumerate(ls[start:], start):
        if pattern in l:
            return i
    return -1

# --- Op 1: Changelog ---
cl_i = find_line(ls, '# ---- Changelog ----')
if cl_i == -1:
    print('ERROR: changelog anchor not found'); sys.exit(1)
if 'punchlist #173' not in ls[cl_i + 1]:
    entry = (
        '# [2026-04-19] CC (punchlist #173) -- TID cascade avoidance (app.py)' + NL +
        '#   What: (b) skip rate-limited fallbacks, (c) cascade_start_ms, (d) cascade metadata in report_outcome' + NL +
        '#   Why:  #173 -- surface cascade depth + wall-time to substrate for learning' + NL +
        '#   How:  readlines patch applied by patch_173_app.py' + NL
    )
    ls.insert(cl_i + 1, entry)

# --- Op 2: Insert _cascade_start_ms before tried_models = [selected_model] ---
tm_i = find_line(ls, 'tried_models = [selected_model]')
if tm_i == -1:
    print('ERROR: tried_models anchor not found'); sys.exit(1)
if '_cascade_start_ms' not in ls[tm_i - 1]:
    ls.insert(tm_i, '    _cascade_start_ms = time.monotonic() * 1000  # cascade wall-time (#173c)' + NL)

# --- Op 3: Insert rate-limit skip before "# Record the failure BEFORE moving on" ---
rec_i = find_line(ls, '# Record the failure BEFORE moving on')
if rec_i == -1:
    print('ERROR: record-failure anchor not found'); sys.exit(1)
if 'is_rate_limited_by_id' not in ls[rec_i - 2]:
    skip_block = (
        '        if _state.model_client and _state.model_client.is_rate_limited_by_id(fallback_id):' + NL +
        '            logger.info("Fallback %s rate-limited -- skipping", fallback_id)' + NL +
        '            continue' + NL +
        NL
    )
    ls.insert(rec_i, skip_block)

# --- Op 4: Add cascade metadata to final report_outcome ---
# Find the final report_outcome (quality_score=quality.overall_score not quality_score=0.0)
qs_i = find_line(ls, 'quality_score=quality.overall_score,')
if qs_i == -1:
    print('ERROR: quality_score anchor not found'); sys.exit(1)
lat_i = qs_i + 1
close_i = lat_i + 1
if 'latency_ms=model_response.latency_ms,' not in ls[lat_i]:
    print('ERROR: latency_ms not where expected, got: ' + ls[lat_i].rstrip()); sys.exit(1)
if 'cascade_depth' not in ls[close_i]:
    if ls[close_i].strip() != ')':
        print('ERROR: closing paren not where expected, got: ' + ls[close_i].rstrip()); sys.exit(1)
    ls[close_i] = (
        '            metadata={' + NL +
        '                "cascade_depth": len(tried_models) - 1,' + NL +
        '                "cascade_total_ms": time.monotonic() * 1000 - _cascade_start_ms,' + NL +
        '                "models_tried": tried_models,' + NL +
        '            },' + NL +
        '        )' + NL
    )

open(p, 'w').write(''.join(ls))
print('APP_PATCHED')
