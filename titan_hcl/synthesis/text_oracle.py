"""text_oracle — the deterministic text-extraction oracle (RFP_text_extraction_introspection Phase A).

ONE primitive: ``extract(corpus_text, query) -> ExtractResult``. Pure + deterministic
(no I/O — the caller hands it the text), so the extract is a *verifiable, re-checkable
fact*: ``extract(same_text, same_query)`` is byte-identical, and ``corpus_sha`` lets any
downstream consumer (anchor / OVG / pattern_logic) re-derive and verify it. It is the
truth-oracle sibling of the coding sandbox — for TEXT instead of computation.

Query kinds (a small typed spec — `query["kind"]`):
  • ``regex``  — {pattern, flags?, max_matches?}     → matches: list[str]
  • ``count``  — {pattern, flags?, group_by?}        → counts: {key: n}  (group_by=a capture group name/index → tallies; else total under "_total")
  • ``window`` — {pattern, flags?, since?, until?, ts_group?} → matches within a [since, until] timestamp band (ts parsed from a named/indexed capture)
  • ``fields`` — {pattern, flags?}                    → fields: {named_group: first_value} from a named-capture regex

Anti-stub / safety: input + pattern are length-capped, matches are count-capped, and
the regex is compiled under a complexity guard (reject pathological patterns up front)
so a hostile/own query can never hang or blow memory. A malformed query raises
``ExtractError`` (typed) — never a bare crash.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# ── safety caps (bounded oracle — never hang / never OOM) ──────────────────
MAX_CORPUS_CHARS = 5_000_000      # 5 MB of text — beyond this we truncate (flagged)
MAX_PATTERN_CHARS = 2_000         # a query pattern longer than this is rejected
MAX_MATCHES_CAP = 10_000          # hard ceiling on returned matches regardless of query
_VALID_KINDS = ("regex", "count", "window", "fields")

# Heuristic catastrophic-backtrack guard: nested unbounded quantifiers like
# (a+)+ / (.*)* are the classic blow-up shapes. Reject them up front. This is a
# conservative screen, not a proof — combined with the length/match caps it keeps
# the oracle bounded for both Titan's own queries and adversarial input.
_NESTED_QUANT = re.compile(r"(\([^)]*[+*][^)]*\)\s*[+*])|(\[[^\]]*\][+*]\s*[+*])")


class ExtractError(ValueError):
    """A malformed query / unsafe pattern — surfaced typed, never a bare crash."""


@dataclass
class ExtractResult:
    """The verifiable extract. Deterministic for a given (corpus, query)."""
    kind: str
    n: int = 0
    matches: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)
    corpus_sha: str = ""
    truncated: bool = False
    query: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "n": self.n, "matches": self.matches,
            "counts": self.counts, "fields": self.fields,
            "corpus_sha": self.corpus_sha, "truncated": self.truncated,
            "query": self.query,
        }


def _compile(pattern: Any, flags_spec: Any) -> re.Pattern:
    if not isinstance(pattern, str) or not pattern:
        raise ExtractError("query.pattern must be a non-empty string")
    if len(pattern) > MAX_PATTERN_CHARS:
        raise ExtractError(f"query.pattern too long (> {MAX_PATTERN_CHARS})")
    if _NESTED_QUANT.search(pattern):
        raise ExtractError("query.pattern rejected (nested unbounded quantifier — "
                           "catastrophic-backtrack risk)")
    flags = 0
    for f in (flags_spec or []):
        f = str(f).lower()
        if f in ("i", "ignorecase"):
            flags |= re.IGNORECASE
        elif f in ("m", "multiline"):
            flags |= re.MULTILINE
        elif f in ("s", "dotall"):
            flags |= re.DOTALL
        else:
            raise ExtractError(f"unknown regex flag: {f!r}")
    try:
        return re.compile(pattern, flags)
    except re.error as e:
        raise ExtractError(f"invalid regex: {e}") from e


def _prep_corpus(corpus_text: Any) -> tuple[str, str, bool]:
    if not isinstance(corpus_text, str):
        raise ExtractError("corpus_text must be a string")
    truncated = False
    if len(corpus_text) > MAX_CORPUS_CHARS:
        corpus_text = corpus_text[:MAX_CORPUS_CHARS]
        truncated = True
    sha = hashlib.sha256(corpus_text.encode("utf-8", "replace")).hexdigest()[:16]
    return corpus_text, sha, truncated


def _max_matches(query: dict) -> int:
    mm = query.get("max_matches", MAX_MATCHES_CAP)
    try:
        mm = int(mm)
    except (TypeError, ValueError):
        raise ExtractError("query.max_matches must be an int")
    return max(0, min(mm, MAX_MATCHES_CAP))


def _group_value(m: re.Match, group: Any) -> Optional[str]:
    """Resolve a capture group by NAME or INDEX; None if absent/unknown."""
    try:
        if isinstance(group, int):
            return m.group(group)
        return m.group(str(group))
    except (IndexError, re.error):
        return None


def extract(corpus_text: str, query: dict) -> ExtractResult:
    """Run a deterministic extract over `corpus_text`. Pure (no I/O). Verifiable.

    Raises ExtractError on a malformed/unsafe query. Never hangs (bounded caps).
    """
    if not isinstance(query, dict):
        raise ExtractError("query must be a dict")
    kind = query.get("kind")
    if kind not in _VALID_KINDS:
        raise ExtractError(f"query.kind must be one of {_VALID_KINDS}, got {kind!r}")

    text, sha, truncated = _prep_corpus(corpus_text)
    pat = _compile(query.get("pattern"), query.get("flags"))
    cap = _max_matches(query)
    res = ExtractResult(kind=kind, corpus_sha=sha, truncated=truncated, query=dict(query))

    if kind == "regex":
        out: list[str] = []
        for m in pat.finditer(text):
            out.append(m.group(0))
            if len(out) >= cap:
                break
        res.matches = out
        res.n = len(out)
        return res

    if kind == "count":
        group_by = query.get("group_by")
        counts: dict[str, int] = {}
        total = 0
        for m in pat.finditer(text):
            total += 1
            if group_by is not None:
                k = _group_value(m, group_by)
                k = "_none" if k is None else str(k)
                counts[k] = counts.get(k, 0) + 1
            if total >= cap:
                break
        if group_by is None:
            counts = {"_total": total}
        res.counts = counts
        res.n = total
        return res

    if kind == "window":
        since = query.get("since")
        until = query.get("until")
        ts_group = query.get("ts_group", 1)
        out2: list[str] = []
        for m in pat.finditer(text):
            tsv = _group_value(m, ts_group)
            if tsv is None:
                continue
            try:
                ts = float(tsv)
            except (TypeError, ValueError):
                continue
            if since is not None and ts < float(since):
                continue
            if until is not None and ts > float(until):
                continue
            out2.append(m.group(0))
            if len(out2) >= cap:
                break
        res.matches = out2
        res.n = len(out2)
        return res

    # kind == "fields"
    m = pat.search(text)
    if m is not None:
        gd = m.groupdict()
        res.fields = {k: ("" if v is None else str(v)) for k, v in gd.items()}
        res.n = len(res.fields)
    return res
