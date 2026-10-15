# ---- Changelog ----
# [2026-10-15] Claude Code (Opus 5.5, lane tid-provider-routing; design 2026-10-15-tid-provider-level-routing §3.A/§3.E,
#   approved by Josh 2026-10-15) — provider facts + hard filters (#1040 / #1126)
# What: ProviderFactsStore — per (model, provider) endpoint facts from OpenRouter's
#   GET /api/v1/models/{author}/{slug}/endpoints (context_length, max_completion_tokens, max_prompt_tokens,
#   quantization, supported_parameters, pricing, uptime, latency, throughput), persisted atomically as msgpack.
#   ProviderFactsRefresher — a daemon thread on its OWN wall-clock timer (LAW 8) that polls the endpoints listing for
#   the models in TID's candidate pool, rate-limited, fail-soft. hard_filter_endpoints() — the "only what cannot
#   work" filter of design §2.3 (context / output cap below what the request needs; deep tier: quantization floor).
# Why: behind one OpenRouter model id sit several providers with different caps / quantization / limits; TID could
#   only see the model. Facts let routing drop endpoints that cannot serve the request and name providers in
#   provider.order / provider.ignore.
# How: same account-key pattern as catalog_manager's /models/user poll (OPENROUTER_API_KEY, Bearer). No key -> no poll
#   (and no network in tests). Stale facts are kept on any failure; a model with no facts gets no hard filter.
#   Endpoint response shape verified against the OpenRouter API reference (list-all-endpoints-for-a-model) and one
#   live read-only GET (qwen/qwen3-coder, 2026-10-15): `tag` is the routing slug ("deepinfra/turbo"), `provider_name`
#   the display name ("DeepInfra") that chat responses report as `provider`.
# -------------------
"""Provider facts (per model, per OpenRouter provider endpoint) and the request hard filters."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("inference_difference.provider_facts")

OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"

# Bits per weight, for the deep-tier quantization floor. Values are OpenRouter's documented quantization levels
# (provider-routing docs, "Quantization"). "unknown" / None are NOT ranked: an unknown quantization is not known to
# be below the floor, so it is never hard-filtered (hard-filter only what cannot work).
_QUANT_BITS = {
    "int4": 4, "fp4": 4, "mxfp4": 4, "nvfp4": 4,
    "fp6": 6,
    "int8": 8, "fp8": 8, "mxfp8": 8,
    "fp16": 16, "bf16": 16,
    "fp32": 32,
}


def quant_bits(q: Optional[str]) -> Optional[int]:
    return _QUANT_BITS.get(str(q).lower()) if q else None


def provider_slug(tag_or_name: Optional[str]) -> str:
    """Base provider slug: 'deepinfra/turbo' -> 'deepinfra'; 'Google Vertex' -> 'google-vertex'."""
    s = (tag_or_name or "").strip().lower()
    if not s:
        return ""
    return s.split("/", 1)[0].replace(" ", "-")


def openrouter_model_name(model_id: str) -> Optional[str]:
    """The OpenRouter model slug for a TID model id, or None if TID would not send it to OpenRouter.

    Reuses model_client._resolve_provider so the answer always matches where the call actually goes.
    """
    try:
        from inference_difference.model_client import _resolve_provider
        base_url, _key, name = _resolve_provider(model_id)
    except Exception:
        return None
    return name if "openrouter.ai" in (base_url or "") else None


@dataclass
class EndpointFact:
    tag: str = ""                     # routing slug, e.g. "deepinfra/turbo"
    slug: str = ""                    # base provider slug, e.g. "deepinfra"
    provider_name: str = ""           # display name, e.g. "DeepInfra" (what chat responses report)
    context_length: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    quantization: Optional[str] = None
    supported_parameters: List[str] = field(default_factory=list)
    pricing: Dict[str, Any] = field(default_factory=dict)
    status: Optional[int] = None
    uptime_last_30m: Optional[float] = None
    uptime_last_1d: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    throughput_p50: Optional[float] = None

    @classmethod
    def from_api(cls, ep: Dict[str, Any]) -> "EndpointFact":
        def _p50(v):
            return v.get("p50") if isinstance(v, dict) else None

        def _int(v):
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        tag = str(ep.get("tag") or "")
        return cls(
            tag=tag,
            slug=provider_slug(tag or ep.get("provider_name")),
            provider_name=str(ep.get("provider_name") or ""),
            context_length=_int(ep.get("context_length")),
            max_completion_tokens=_int(ep.get("max_completion_tokens")),
            max_prompt_tokens=_int(ep.get("max_prompt_tokens")),
            quantization=(str(ep["quantization"]) if ep.get("quantization") else None),
            supported_parameters=[str(x) for x in (ep.get("supported_parameters") or [])],
            pricing=dict(ep.get("pricing") or {}),
            status=_int(ep.get("status")),
            uptime_last_30m=ep.get("uptime_last_30m"),
            uptime_last_1d=ep.get("uptime_last_1d"),
            latency_p50_ms=_p50(ep.get("latency_last_30m")),
            throughput_p50=_p50(ep.get("throughput_last_30m")),
        )


@dataclass
class RequestNeeds:
    """What one request needs from an endpoint (inputs to the hard filter)."""
    context_tokens: int = 0           # total window needed (prompt + output), before margin
    prompt_tokens: int = 0            # prompt only, before margin
    output_tokens: int = 0            # output cap needed, before margin
    deep: bool = False                # deep-reasoning tier -> quantization floor + minimum output cap


def endpoint_fails(ep: EndpointFact, needs: RequestNeeds, *, margin: float, quant_floor: str,
                   deep_min_output: int) -> Optional[str]:
    """Reason string if this endpoint cannot serve the request, else None. Unknown caps never fail."""
    m = 1.0 + max(0.0, float(margin))
    if ep.context_length and needs.context_tokens and ep.context_length < needs.context_tokens * m:
        return f"context {ep.context_length} < {int(needs.context_tokens * m)}"
    if ep.max_prompt_tokens and needs.prompt_tokens and ep.max_prompt_tokens < needs.prompt_tokens * m:
        return f"prompt cap {ep.max_prompt_tokens} < {int(needs.prompt_tokens * m)}"
    out_need = needs.output_tokens * m if needs.output_tokens else 0
    if needs.deep:
        out_need = max(out_need, float(deep_min_output or 0))
    if ep.max_completion_tokens and out_need and ep.max_completion_tokens < out_need:
        return f"output cap {ep.max_completion_tokens} < {int(out_need)}"
    if needs.deep:
        floor = quant_bits(quant_floor)
        have = quant_bits(ep.quantization)
        if floor is not None and have is not None and have < floor:
            return f"quantization {ep.quantization} < {quant_floor}"
    return None


def hard_filter_endpoints(endpoints: Iterable[EndpointFact], needs: RequestNeeds, *, margin: float,
                          quant_floor: str, deep_min_output: int
                          ) -> Tuple[List[EndpointFact], List[Tuple[EndpointFact, str]]]:
    """Split endpoints into (usable, [(dropped, reason)])."""
    keep: List[EndpointFact] = []
    drop: List[Tuple[EndpointFact, str]] = []
    for ep in endpoints:
        why = endpoint_fails(ep, needs, margin=margin, quant_floor=quant_floor, deep_min_output=deep_min_output)
        (drop.append((ep, why)) if why else keep.append(ep))
    return keep, drop


class ProviderFactsStore:
    """Thread-safe cache of endpoint facts per TID model id, persisted atomically (msgpack)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._models: Dict[str, Dict[str, Any]] = {}   # model_id -> {"fetched_at": ts, "endpoints": [EndpointFact]}
        self.last_sweep_at: float = 0.0
        self.last_error: str = ""

    # ----- queries -----
    def endpoints(self, model_id: str) -> List[EndpointFact]:
        with self._lock:
            e = self._models.get(model_id)
            return list(e["endpoints"]) if e else []

    def has(self, model_id: str) -> bool:
        with self._lock:
            return bool(self._models.get(model_id, {}).get("endpoints"))

    def slugs(self, model_id: str) -> List[str]:
        seen: List[str] = []
        for ep in self.endpoints(model_id):
            if ep.slug and ep.slug not in seen:
                seen.append(ep.slug)
        return seen

    def known_slugs(self) -> set:
        with self._lock:
            return {ep.slug for e in self._models.values() for ep in e["endpoints"] if ep.slug}

    def slug_for_name(self, name: Optional[str], model_id: Optional[str] = None) -> str:
        """Map a served-provider display name (or slug/tag) to a base slug, preferring this model's facts."""
        if not name:
            return ""
        low = str(name).strip().lower()
        with self._lock:
            pools = []
            if model_id and model_id in self._models:
                pools.append(self._models[model_id]["endpoints"])
            pools.extend(e["endpoints"] for e in self._models.values())
            for eps in pools:
                for ep in eps:
                    if low in (ep.provider_name.lower(), ep.tag.lower(), ep.slug):
                        return ep.slug
        return provider_slug(name)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n_eps = sum(len(e["endpoints"]) for e in self._models.values())
            return {"models": len(self._models), "endpoints": n_eps,
                    "last_sweep_at": self.last_sweep_at, "last_error": self.last_error}

    # ----- mutation -----
    def put(self, model_id: str, endpoints: List[EndpointFact], fetched_at: Optional[float] = None) -> None:
        with self._lock:
            self._models[model_id] = {"fetched_at": fetched_at or time.time(), "endpoints": list(endpoints)}

    def fetched_at(self, model_id: str) -> float:
        with self._lock:
            return float(self._models.get(model_id, {}).get("fetched_at", 0.0))

    # ----- persistence -----
    def save(self) -> None:
        import msgpack
        from inference_difference.atomic_io import atomic_write_bytes
        with self._lock:
            payload = {
                "saved_at": time.time(),
                "last_sweep_at": self.last_sweep_at,
                "models": {mid: {"fetched_at": e["fetched_at"], "endpoints": [asdict(ep) for ep in e["endpoints"]]}
                           for mid, e in self._models.items()},
            }
        atomic_write_bytes(self.path, msgpack.packb(payload))

    def load(self) -> None:
        """Load the persisted facts. Absent / corrupt file -> start empty (no facts = no hard filter)."""
        import msgpack
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "rb") as f:
                data = msgpack.unpackb(f.read(), raw=False)
            models: Dict[str, Dict[str, Any]] = {}
            for mid, e in (data.get("models") or {}).items():
                eps = []
                for d in e.get("endpoints") or []:
                    try:
                        eps.append(EndpointFact(**{k: v for k, v in d.items()
                                                   if k in EndpointFact.__dataclass_fields__}))
                    except TypeError:
                        continue
                models[str(mid)] = {"fetched_at": float(e.get("fetched_at", 0.0)), "endpoints": eps}
            with self._lock:
                self._models = models
                self.last_sweep_at = float(data.get("last_sweep_at", 0.0))
            logger.info("Provider facts loaded: %d models", len(models))
        except Exception as e:  # noqa: BLE001 — fail-soft
            logger.warning("Provider facts load failed (starting empty): %s", e)


def fetch_endpoints_openrouter(or_model: str, api_key: str, timeout: float = 20.0) -> List[EndpointFact]:
    """GET the endpoints listing for one OpenRouter model. Raises on transport / HTTP failure."""
    import httpx
    resp = httpx.get(OPENROUTER_ENDPOINTS_URL.format(model=or_model), timeout=timeout,
                     headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"})
    resp.raise_for_status()
    data = (resp.json() or {}).get("data") or {}
    return [EndpointFact.from_api(ep) for ep in (data.get("endpoints") or []) if isinstance(ep, dict)]


class ProviderFactsRefresher:
    """Own-clock poller (LAW 8): wakes on a wall-clock interval, whether or not anyone is talking.

    One sweep = for each OpenRouter model in the candidate pool, fetch its endpoints listing, spaced by
    ``min_spacing_s`` (rate limit). Least-recently-fetched models go first so a sweep cut short by shutdown still
    makes progress. Every failure is logged and skipped; the previous facts for that model are kept (fail-soft).
    """

    def __init__(self, store: ProviderFactsStore, models_fn: Callable[[], List[str]], *,
                 interval_s: float = 3600.0, initial_delay_s: float = 20.0, min_spacing_s: float = 1.0,
                 fetch_fn: Optional[Callable[[str, str], List[EndpointFact]]] = None,
                 api_key_fn: Optional[Callable[[], str]] = None, fetch_timeout_s: float = 20.0) -> None:
        self.store = store
        self._models_fn = models_fn
        self._interval = max(1.0, float(interval_s))
        self._initial_delay = max(0.0, float(initial_delay_s))
        self._spacing = max(0.0, float(min_spacing_s))
        self._fetch = fetch_fn or (lambda m, k: fetch_endpoints_openrouter(m, k, timeout=fetch_timeout_s))
        self._api_key_fn = api_key_fn or (lambda: os.environ.get("OPENROUTER_API_KEY", ""))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.sweeps = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tid-provider-facts", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        if self._stop.wait(self._initial_delay):
            return
        while not self._stop.is_set():
            try:
                self.sweep_once()
            except Exception as e:  # noqa: BLE001 — the loop must never die
                logger.warning("Provider facts sweep failed: %s", e)
            if self._stop.wait(self._interval):
                return

    def sweep_once(self) -> int:
        """One rate-limited pass. Returns the number of models refreshed."""
        api_key = self._api_key_fn() or ""
        if not api_key:
            logger.debug("Provider facts: no OPENROUTER_API_KEY — sweep skipped")
            return 0
        pairs = []
        for mid in self._models_fn() or []:
            or_name = openrouter_model_name(mid)
            if or_name:
                pairs.append((self.store.fetched_at(mid), mid, or_name))
        pairs.sort()
        done = 0
        for i, (_ts, mid, or_name) in enumerate(pairs):
            if self._stop.is_set():
                break
            if i and self._spacing and self._stop.wait(self._spacing):
                break
            try:
                eps = self._fetch(or_name, api_key)
                self.store.put(mid, eps)
                done += 1
            except Exception as e:  # noqa: BLE001 — keep stale facts
                self.store.last_error = f"{or_name}: {str(e)[:160]}"
                logger.debug("Provider facts fetch failed for %s (kept stale): %s", or_name, e)
        self.store.last_sweep_at = time.time()
        self.sweeps += 1
        try:
            self.store.save()
        except Exception as e:  # noqa: BLE001
            logger.warning("Provider facts save failed: %s", e)
        logger.info("Provider facts sweep: %d/%d models refreshed", done, len(pairs))
        return done
