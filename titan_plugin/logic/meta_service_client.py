"""
Meta-Reasoning Consumer Service — shared client helper.

Public API used by all 8 CGN consumers (social, language, knowledge, reasoning,
coding, self_model, emotional, dreaming) to invoke meta-reasoning as a service.

See rFP: titan-docs/rFP_meta_service_interface.md §4 + §16.0.

Three public functions:
    send_meta_request(...)       → str request_id    (emit META_REASON_REQUEST)
    register_response_handler(...) → None            (bind consumer → callable)
    send_meta_outcome(...)       → None              (emit META_REASON_OUTCOME)

Plus one dispatch helper used by worker receive loops:
    dispatch_meta_response(msg, logger=None) → bool

The helper itself never touches the bus directly — it puts messages on the
consumer's `send_queue` (multiprocessing.Queue), which Guardian drains into
the parent process's bus. This matches the existing worker pattern for
META_CGN_SIGNAL, CGN_TRANSITION, META_LANGUAGE_REQUEST.

Schema validation happens at send time so bugs surface at the caller, not
silently drop on the receiving end.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ── Schema constants (stable contract — keep in sync with rFP §4) ────

# Question-type enum (rFP §4.2). Meta-service rejects unknown types at the
# aggregator so typos here surface immediately.
KNOWN_QUESTION_TYPES = frozenset({
    "formulate_strategy",
    "recall_context",
    "evaluate_option",
    "hypothesize_cause",
    "synthesize_insight",
    "break_impasse",
    "introspect_state",
    "spirit_self_nudge",
})

# Consumer IDs (rFP §4.6). Must match [meta_service_interface.consumer_home_worker]
# keys in titan_params.toml — validator checks both in tests.
KNOWN_CONSUMERS = frozenset({
    "social",
    "language",
    "knowledge",
    "reasoning",
    "coding",
    "self_model",
    "emotional",
    "dreaming",
    "reflection",
})

# Failure modes returned in META_REASON_RESPONSE.failure_mode (rFP §4.1).
FAILURE_MODES = frozenset({
    "timeout_budget",
    "low_confidence",
    "rate_limited",
    "recursion_cap",
    "impasse_hit",
    "not_yet_implemented",  # Session 1 dry-run handler
})

# Context vector must be exactly this wide (rFP §4.1 + §16).
CONTEXT_VECTOR_DIM = 30

# Outcome reward hard bounds (rFP §4.6).
OUTCOME_REWARD_MIN = -1.0
OUTCOME_REWARD_MAX = 1.0

# Bus message type constants (mirrored here so consumers can import in one
# place without needing to know bus.py layout). Must stay identical.
MSG_META_REASON_REQUEST = "META_REASON_REQUEST"
MSG_META_REASON_RESPONSE = "META_REASON_RESPONSE"
MSG_META_REASON_OUTCOME = "META_REASON_OUTCOME"

# ── Process-local state — handlers registered per-consumer ───────────
# Each worker process owns its own copy of this dict (Python import cache
# gives each worker its own module instance). register_response_handler()
# populates it; dispatch_meta_response() consumes it.
_response_handlers: Dict[str, Callable[[dict], None]] = {}


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def send_meta_request(
    consumer_id: str,
    question_type: str,
    context_vector: list,
    time_budget_ms: int,
    constraints: Optional[dict] = None,
    payload_snippet: str = "",
    send_queue=None,
    src: str = "",
) -> str:
    """Emit META_REASON_REQUEST via the consumer's send_queue.

    Returns the request_id (UUID4 str). Consumer stores it to correlate
    the upcoming META_REASON_RESPONSE.

    Raises ValueError on schema violation so bugs surface at the caller.

    Args:
        consumer_id:     Must be one of KNOWN_CONSUMERS.
        question_type:   Must be one of KNOWN_QUESTION_TYPES.
        context_vector:  List of 30 floats (CONTEXT_VECTOR_DIM).
        time_budget_ms:  MANDATORY per rFP §4.7 — no default. Consumer
                         decides its own latency tolerance per-request.
        constraints:     Optional dict — keys: confidence_threshold,
                         max_chain_length, allow_timechain_query.
        payload_snippet: Optional human-readable tag for debugging.
        send_queue:      multiprocessing.Queue of the caller's worker.
                         If None, caller must put the returned message
                         on the bus directly.
        src:             Source subscriber name (e.g. "spirit", "language").
                         Required when send_queue is provided.

    Returns:
        request_id (str) — UUID4.
    """
    if consumer_id not in KNOWN_CONSUMERS:
        raise ValueError(
            f"meta_service_client: unknown consumer_id={consumer_id!r}, "
            f"expected one of {sorted(KNOWN_CONSUMERS)}")
    if question_type not in KNOWN_QUESTION_TYPES:
        raise ValueError(
            f"meta_service_client: unknown question_type={question_type!r}, "
            f"expected one of {sorted(KNOWN_QUESTION_TYPES)}")
    if not isinstance(context_vector, list) or len(context_vector) != CONTEXT_VECTOR_DIM:
        raise ValueError(
            f"meta_service_client: context_vector must be list of length "
            f"{CONTEXT_VECTOR_DIM}, got {type(context_vector).__name__} "
            f"len={len(context_vector) if hasattr(context_vector, '__len__') else '?'}")
    if not isinstance(time_budget_ms, int) or time_budget_ms <= 0:
        raise ValueError(
            f"meta_service_client: time_budget_ms must be positive int, got "
            f"{time_budget_ms!r} (MANDATORY per rFP §4.7 — no default)")

    request_id = str(uuid.uuid4())
    payload = {
        "consumer_id": consumer_id,
        "question_type": question_type,
        "context_vector": list(context_vector),
        "time_budget_ms": int(time_budget_ms),
        "constraints": dict(constraints) if constraints else {},
        "payload_snippet": str(payload_snippet)[:256],  # cap noise
        "request_id": request_id,
    }
    msg = {
        "type": MSG_META_REASON_REQUEST,
        "src": src or consumer_id,
        "dst": "spirit",  # meta_service lives in spirit_worker
        "ts": time.time(),
        "rid": None,
        "payload": payload,
    }
    if send_queue is not None:
        try:
            send_queue.put_nowait(msg)
        except Exception as e:
            logger.warning(
                "[meta_service_client] send_queue.put_nowait failed for "
                "%s.%s: %s (request_id=%s)",
                consumer_id, question_type, e, request_id)
    return request_id


def register_response_handler(
    consumer_id: str,
    handler_fn: Callable[[dict], None],
) -> None:
    """Bind handler_fn to consumer_id. Called once during consumer init.

    handler_fn receives the META_REASON_RESPONSE payload dict (not the full
    bus message). It decides what to do with insight.suggested_action.

    Re-registering overwrites the previous handler — callers assume at-most-
    one handler per consumer_id per process.
    """
    if consumer_id not in KNOWN_CONSUMERS:
        raise ValueError(
            f"meta_service_client: unknown consumer_id={consumer_id!r}")
    if not callable(handler_fn):
        raise ValueError(
            f"meta_service_client: handler_fn must be callable, got "
            f"{type(handler_fn).__name__}")
    _response_handlers[consumer_id] = handler_fn
    logger.info(
        "[meta_service_client] response handler registered for consumer='%s'",
        consumer_id)


def send_meta_outcome(
    request_id: str,
    consumer_id: str,
    outcome_reward: float,
    actual_primitive_used: Optional[str] = None,
    context: str = "",
    send_queue=None,
    src: str = "",
) -> None:
    """Emit META_REASON_OUTCOME via the consumer's send_queue.

    Called by the consumer after it has observed whether meta's insight
    actually helped. outcome_reward is SIGNED per Maker decision (rFP §4.6):

        +1.0 = advice clearly right
         0.0 = advice didn't help (null)
        -1.0 = advice clearly wrong  ← teaches meta what NOT to do

    Raises ValueError on out-of-range outcome_reward.
    """
    if consumer_id not in KNOWN_CONSUMERS:
        raise ValueError(
            f"meta_service_client: unknown consumer_id={consumer_id!r}")
    if not isinstance(outcome_reward, (int, float)):
        raise ValueError(
            f"meta_service_client: outcome_reward must be float, got "
            f"{type(outcome_reward).__name__}")
    if not (OUTCOME_REWARD_MIN <= outcome_reward <= OUTCOME_REWARD_MAX):
        raise ValueError(
            f"meta_service_client: outcome_reward must be in "
            f"[{OUTCOME_REWARD_MIN}, {OUTCOME_REWARD_MAX}], got "
            f"{outcome_reward}")
    if not request_id:
        raise ValueError("meta_service_client: request_id is required")

    payload = {
        "consumer_id": consumer_id,
        "request_id": request_id,
        "outcome_reward": float(outcome_reward),
        "actual_primitive_used": actual_primitive_used,
        "context": str(context)[:256],
    }
    msg = {
        "type": MSG_META_REASON_OUTCOME,
        "src": src or consumer_id,
        "dst": "spirit",
        "ts": time.time(),
        "rid": None,
        "payload": payload,
    }
    if send_queue is not None:
        try:
            send_queue.put_nowait(msg)
        except Exception as e:
            logger.warning(
                "[meta_service_client] outcome send failed for %s "
                "(request_id=%s): %s",
                consumer_id, request_id, e)


# ─────────────────────────────────────────────────────────────────────
# Dispatch helper — called by worker receive loops when a
# META_REASON_RESPONSE lands on their queue.
# ─────────────────────────────────────────────────────────────────────

def dispatch_meta_response(msg: dict, logger_obj=None) -> bool:
    """Dispatch incoming META_REASON_RESPONSE to the registered handler.

    Returns True if a handler was invoked, False if no handler was registered
    for the consumer_id in the payload. Workers should call this from their
    main receive loop on msg_type == "META_REASON_RESPONSE".

    Never raises — handler exceptions are caught and logged. A failing
    handler must NOT take down the worker.
    """
    log = logger_obj or logger
    if not isinstance(msg, dict):
        log.warning("[meta_service_client] dispatch: non-dict msg=%r", msg)
        return False
    payload = msg.get("payload") or {}
    consumer_id = payload.get("consumer_id", "")
    handler = _response_handlers.get(consumer_id)
    if handler is None:
        log.debug(
            "[meta_service_client] no handler registered for consumer='%s' "
            "(request_id=%s) — response dropped",
            consumer_id, payload.get("request_id", "?"))
        return False
    try:
        handler(payload)
        return True
    except Exception as e:
        log.warning(
            "[meta_service_client] handler for consumer='%s' raised "
            "%s: %s (request_id=%s)",
            consumer_id, type(e).__name__, e, payload.get("request_id", "?"))
        return False


# ─────────────────────────────────────────────────────────────────────
# Introspection helpers (used by /v4/meta-service endpoint)
# ─────────────────────────────────────────────────────────────────────

def get_registered_consumers() -> list:
    """Return consumer_ids with active local handlers. Diagnostic only."""
    return sorted(_response_handlers.keys())


def _clear_handlers_for_testing() -> None:
    """Test-only: reset process-local handler registry. Never call from prod."""
    _response_handlers.clear()
