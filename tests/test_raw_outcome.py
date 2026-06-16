from inference_difference.outcome import raw_outcome

def test_usable_content_is_success():
    ok, strength, meta = raw_outcome(call_ok=True, content="Here is an answer.",
                                     tool_calls=None, tools_requested=False, latency_ms=8000.0)
    assert ok is True and strength == 1.0
    assert meta["had_content"] is True and meta["had_tool_calls"] is False
    assert meta["tools_requested"] is False and meta["latency_ms"] == 8000.0

def test_tool_calls_only_is_success():
    ok, _, meta = raw_outcome(call_ok=True, content="", tool_calls=[{"id": "x"}],
                              tools_requested=True, latency_ms=100.0)
    assert ok is True and meta["had_tool_calls"] is True

def test_empty_response_is_failure():
    ok, _, _ = raw_outcome(call_ok=True, content="   ", tool_calls=None,
                           tools_requested=False, latency_ms=50.0)
    assert ok is False

def test_call_not_ok_is_failure():
    ok, _, _ = raw_outcome(call_ok=False, content="partial", tool_calls=None,
                           tools_requested=False, latency_ms=50.0)
    assert ok is False

def test_slow_usable_response_is_still_success():
    # THE #330 regression: a usable response with huge latency must NOT be a failure.
    ok, strength, _ = raw_outcome(call_ok=True, content="A thorough, correct answer.",
                                  tool_calls=None, tools_requested=False, latency_ms=99999.0)
    assert ok is True and strength == 1.0


def test_nonstreaming_deposit_uses_raw_success():
    # The deposit keys on raw usability, NOT quality.is_success: slow+usable -> success.
    ok, strength, meta = raw_outcome(call_ok=True, content="answer", tool_calls=None,
                                     tools_requested=True, latency_ms=90000.0)
    assert ok is True
    assert meta["tools_requested"] is True
