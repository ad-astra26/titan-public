"""
titan_plugin/logic/language_config.py — Per-Titan Language Configuration.

Reads [language] base config + [language.T1]/[language.T2]/[language.T3]
overrides from titan_params.toml. Returns a flat merged dict for the
current Titan's identity.

Used by both spirit_worker (Phase 0) and language_worker (Phase 1+).
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Defaults — used if titan_params.toml section is missing
DEFAULTS = {
    "enabled": True,
    "teacher_interval_compositions": 5,
    "bootstrap_vocab_threshold": 50,
    "exploration_da_threshold": "setpoint",
    "conversation_temperature": 0.7,
    "max_acquisition_per_session": 3,
    "composition_max_level": 8,
    "teaching_timeout_s": 90,
    "conversation_timeout_s": 600,
    "teacher_queue_max": 10,
    "bootstrap_cooldown_s": 60,
    "bootstrap_retry_threshold": 3,
}


def _get_titan_id() -> str:
    """Read titan_id from data/titan_identity.json."""
    try:
        path = os.path.join(_PROJECT_ROOT, "data", "titan_identity.json")
        with open(path) as f:
            return json.load(f).get("titan_id", "T1")
    except Exception:
        return "T1"


def load_config(titan_id: str | None = None) -> dict:
    """Load language config with per-Titan overrides.

    Args:
        titan_id: Override Titan identity (default: read from titan_identity.json)

    Returns:
        Flat dict with all resolved language config values.
    """
    if titan_id is None:
        titan_id = _get_titan_id()

    # Read titan_params.toml
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # Python < 3.11

    params_path = os.path.join(
        _PROJECT_ROOT, "titan_plugin", "titan_params.toml")

    base = dict(DEFAULTS)

    try:
        with open(params_path, "rb") as f:
            params = tomllib.load(f)

        # Merge [language] base section
        lang_section = params.get("language", {})
        for k, v in lang_section.items():
            if not isinstance(v, dict):  # Skip sub-tables (T1/T2/T3)
                base[k] = v

        # Merge per-Titan overrides
        titan_overrides = lang_section.get(titan_id, {})
        for k, v in titan_overrides.items():
            base[k] = v

    except FileNotFoundError:
        logger.warning("[LanguageConfig] titan_params.toml not found, using defaults")
    except Exception as e:
        logger.warning("[LanguageConfig] Failed to load config: %s", e)

    base["titan_id"] = titan_id
    return base
