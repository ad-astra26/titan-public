"""Felt-at-lived-time snapshot fix (RFP_synthesis_engram_grounding §7.C).

The PreHook's pre-chat neuromod snapshot — the source of every promoted chat thought's
felt (→ Engram axis_felt → synthesis felt-bridge / CGN-felt frame_dependent) — was
sourced from the agno coordinator, whose `neuromodulators` is EMPTY in the agno PreHook
path. Result: `felt=None` on 100% of turns, `axis_felt=0` fleet-wide. Fix:
`_capture_pre_chat_felt` sources from SHM `read_neuromod()` (the authoritative live slot,
`{modulator: {"level": float}}`) and runs for every tier.

Run: python -m pytest tests/test_agno_felt_snapshot_20260605.py -v -p no:anchorpy
"""
import warnings
warnings.filterwarnings("ignore")

from titan_hcl.modules.agno_hooks import _capture_pre_chat_felt


class _Plugin:
    """Minimal stand-in carrying a pre-seeded SHM bank (so the helper doesn't
    construct a real ShmReaderBank in the test)."""
    def __init__(self, bank):
        self._shm_reader_bank = bank


class _Bank:
    def __init__(self, payload):
        self._payload = payload

    def read_neuromod(self):
        return self._payload


def test_dict_shaped_modulators_extract_levels():
    # The real read_neuromod() shape: {modulator: {"level": float}}.
    bank = _Bank({"modulators": {
        "DA": {"level": 0.80}, "5-HT": {"level": 0.30},
        "NE": {"level": 0.75}, "GABA": {"level": 0.12},
    }})
    p = _Plugin(bank)
    snap = _capture_pre_chat_felt(p)
    assert snap == {"DA": 0.80, "5-HT": 0.30, "NE": 0.75, "GABA": 0.12}
    assert p._pre_chat_neuromods == snap  # stored on the plugin


def test_real_modulators_are_non_neutral():
    # The whole point: real lived levels, NOT collapsed to neutral 0.5.
    bank = _Bank({"modulators": {"DA": {"level": 0.9}}})
    snap = _capture_pre_chat_felt(_Plugin(bank))
    assert snap["DA"] == 0.9 and snap["DA"] != 0.5


def test_float_shaped_modulators_defensive():
    # Defensive: if a source ever hands floats, use them directly (not 0.5).
    bank = _Bank({"modulators": {"DA": 0.7, "NE": 0.4}})
    snap = _capture_pre_chat_felt(_Plugin(bank))
    assert snap == {"DA": 0.7, "NE": 0.4}


def test_empty_modulators_yields_empty_snapshot():
    assert _capture_pre_chat_felt(_Plugin(_Bank({"modulators": {}}))) == {}


def test_none_read_yields_empty():
    assert _capture_pre_chat_felt(_Plugin(_Bank(None))) == {}


def test_missing_modulators_key_yields_empty():
    assert _capture_pre_chat_felt(_Plugin(_Bank({"age_seconds": 1.0}))) == {}


def test_bank_raises_soft_fails_to_empty():
    class _BoomBank:
        def read_neuromod(self):
            raise RuntimeError("shm down")
    p = _Plugin(_BoomBank())
    assert _capture_pre_chat_felt(p) == {}
    assert p._pre_chat_neuromods == {}  # still set (never AttributeError downstream)


def test_bool_values_excluded():
    # bool is an int subclass — must not leak in as 1.0/0.0.
    bank = _Bank({"modulators": {"DA": True, "NE": {"level": 0.6}}})
    assert _capture_pre_chat_felt(_Plugin(bank)) == {"NE": 0.6}


if __name__ == "__main__":
    for fn in (test_dict_shaped_modulators_extract_levels,
               test_real_modulators_are_non_neutral,
               test_float_shaped_modulators_defensive,
               test_empty_modulators_yields_empty_snapshot,
               test_none_read_yields_empty,
               test_missing_modulators_key_yields_empty,
               test_bank_raises_soft_fails_to_empty,
               test_bool_values_excluded):
        fn()
    print("OK — felt snapshot SHM-source fix verified")
