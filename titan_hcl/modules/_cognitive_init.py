"""Cognitive engine init helpers — extracted from spirit_worker.py.

Per chunk 8C of PLAN_microkernel_phase_c_s8_cognitive_worker_extraction.md.

Pure relocation — no behavior change. The four init helpers below were
introduced in spirit_worker.py during the L3 cognitive engine bring-up
(InnerTrinityCoordinator + ObservableEngine + NeuralNervousSystem +
T2 state registries). Chunk 8E will add cognitive_worker.py which
imports these same helpers for the l0_rust_enabled=true path; the
legacy spirit_worker_main path (l0_rust_enabled=false) continues to
import them too. The shared module ensures one canonical implementation
honors G17 (one crate per concept — applied to Python by analogy).

All helpers:
    - Return None on failure (engines optional / lazy / per-Titan capability)
    - Do all heavy imports lazily inside the function (per
      feedback_lazy_imports_titan_hcl.md — heavy imports MUST be lazy)
    - Use the [SpiritWorker] log prefix unchanged for chunk 8C (no behavior
      change). Chunk 8E will genericize the prefix when cognitive_worker
      starts calling these helpers.

Path math note: `_cognitive_init.py` lives in the same directory as
`spirit_worker.py` (titan_hcl/modules/). All `os.path.dirname(__file__)`
expressions resolve identically — the relocation is bytewise behavior-
preserving for `titan_params.toml` lookup + data_dir resolution.
"""
from __future__ import annotations

import logging
import os
from titan_hcl.params import get_params

logger = logging.getLogger(__name__)


def _init_observable_engine():
    """Initialize T1 Observable Engine (5 observables × 6 body parts)."""
    try:
        from titan_hcl.logic.observables import ObservableEngine
        return ObservableEngine()
    except Exception as e:
        logger.warning("[SpiritWorker] ObservableEngine init failed: %s", e)
        return None


def _init_neural_nervous_system(config: dict):
    """Initialize V5 Neural Nervous System (config-driven, learned reflexes)."""
    try:
        # RFP_config_as_shm_state §7.C/C.3b: read [neural_nervous_system] from
        # the SHM slot (config-as-state, INV-CFG-7).
        params_config = {}
        try:
            params_config = get_params("neural_nervous_system")
        except Exception:
            pass

        if not params_config.get("enabled", False):
            return None

        from titan_hcl.logic.neural_nervous_system import NeuralNervousSystem
        # V4 VM NervousSystem as fallback
        vm_ns = None
        try:
            from titan_hcl.logic.nervous_system import NervousSystem
            vm_ns = NervousSystem(config=get_params("titan_vm"))
        except Exception:
            pass

        data_dir = config.get("data_dir", "./data")
        if not data_dir:
            data_dir = "./data"
        nn_data_dir = os.path.join(data_dir, "neural_nervous_system")

        return NeuralNervousSystem(
            config=params_config,
            data_dir=nn_data_dir,
            vm_nervous_system=vm_ns,
        )
    except Exception as e:
        logger.warning("[SpiritWorker] NeuralNervousSystem init failed: %s", e)
        return None


def _init_coordinator(inner_state, spirit_state, observable_engine,
                      neural_nervous_system=None, config: dict | None = None):
    """Initialize T3 Inner Trinity Coordinator with T4/V5 Nervous System.

    ``config`` (the full spirit worker config dict) is threaded through so
    the [titan_vm] toml section reaches the lightweight T4 VM. Added
    2026-04-16 with the [titan_vm] plumb fix.
    """
    try:
        from titan_hcl.logic.inner_coordinator import InnerTrinityCoordinator
        # T4: Create NervousSystem with lightweight TitanVM (context-only)
        nervous_system = None
        try:
            from titan_hcl.logic.nervous_system import NervousSystem
            vm_cfg = get_params("titan_vm")
            nervous_system = NervousSystem(config=vm_cfg)
        except Exception as e:
            logger.warning("[SpiritWorker] NervousSystem init failed: %s", e)
        # T5: Create TopologyEngine
        topology_engine = None
        try:
            from titan_hcl.logic.topology import TopologyEngine
            topology_engine = TopologyEngine()
        except Exception as e:
            logger.warning("[SpiritWorker] TopologyEngine init failed: %s", e)
        # T6: Create DreamingEngine
        dreaming_engine = None
        try:
            from titan_hcl.logic.dreaming import DreamingEngine
            _dreaming_state_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "dreaming_state.json")
            # Load DNA weights from titan_params.toml [dreaming]
            _dreaming_dna = {}
            try:
                import tomllib as _tl
                _tp = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "titan_params.toml")
                if os.path.exists(_tp):
                    with open(_tp, "rb") as _tf:
                        _dreaming_dna = _tl.load(_tf).get("dreaming", {})
            except Exception:
                pass
            dreaming_engine = DreamingEngine(
                state_path=_dreaming_state_path, dna=_dreaming_dna)
        except Exception as e:
            logger.warning("[SpiritWorker] DreamingEngine init failed: %s", e)
        return InnerTrinityCoordinator(
            inner_state=inner_state,
            spirit_state=spirit_state,
            observable_engine=observable_engine,
            vm=None,
            nervous_system=nervous_system,
            topology_engine=topology_engine,
            dreaming_engine=dreaming_engine,
            neural_nervous_system=neural_nervous_system,
        )
    except Exception as e:
        logger.warning("[SpiritWorker] InnerTrinityCoordinator init failed: %s", e)
        return None


def _init_t2_state_registries():
    """Initialize T2 InnerState + SpiritState registries."""
    inner, spirit = None, None
    try:
        from titan_hcl.logic.inner_state import InnerState
        inner = InnerState()
    except Exception as e:
        logger.warning("[SpiritWorker] InnerState init failed: %s", e)
    try:
        from titan_hcl.logic.spirit_state import SpiritState
        spirit = SpiritState()
    except Exception as e:
        logger.warning("[SpiritWorker] SpiritState init failed: %s", e)
    return inner, spirit
