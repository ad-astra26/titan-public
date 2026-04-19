"""
TimeChain Worker — Guardian-supervised module for Proof of Thought memory chain.

Centralized write path for TimeChain blocks. All memory sources submit thoughts
via bus messages; this worker validates with Proof of Thought and commits to
the append-only hash chain.

Bus messages consumed:
  - TIMECHAIN_COMMIT: Submit a thought for PoT validation + chain commit
  - TIMECHAIN_QUERY: Query blocks by type/source/fork/tag/epoch
  - TIMECHAIN_STATUS: Request chain status
  - EPOCH_TICK: Periodic heartbeat blocks on main chain
  - DREAMING_START/END: Dream lifecycle events → episodic blocks
  - MODULE_SHUTDOWN: Clean shutdown
  - QUERY: Standard query handler for arch_map integration

Bus messages produced:
  - TIMECHAIN_COMMITTED: Block successfully committed (with block_hash)
  - TIMECHAIN_REJECTED: Block rejected by PoT (with reason)
  - TIMECHAIN_CHECKPOINT: Merkle checkpoint created
"""

import json
import logging
import os
import sys
import time

logger = logging.getLogger("TimeChainWorker")


def timechain_worker_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main loop for the TimeChain module process."""
    from queue import Empty

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ── Heartbeat helper (Phase E Fix 2: throttled to 3s min interval) ──
    _hb_state = {"last": 0.0}

    def _send_heartbeat():
        now = time.time()
        if now - _hb_state["last"] < 3.0:
            return
        _hb_state["last"] = now
        try:
            import psutil
            rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            rss_mb = 0
        send_queue.put({"type": "MODULE_HEARTBEAT", "src": name, "dst": "guardian",
                        "payload": {"rss_mb": round(rss_mb, 1)}, "ts": time.time()})

    def _send_msg(msg_type, dst, payload):
        send_queue.put({"type": msg_type, "src": name, "dst": dst,
                        "payload": payload, "ts": time.time()})

    _send_heartbeat()
    logger.info("[TimeChain] Initializing TimeChain worker...")

    # ── Load titan identity ──
    titan_id = "T1"
    try:
        with open("./data/titan_identity.json") as f:
            titan_id = json.load(f).get("titan_id", "T1")
    except Exception:
        pass

    # ── Load config from titan_params.toml ──
    tc_config = {}
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open("titan_plugin/titan_params.toml", "rb") as f:
            all_params = tomllib.load(f)
        tc_config = all_params.get("timechain", {})
    except Exception as e:
        logger.warning("[TimeChain] Could not load titan_params.toml: %s", e)

    enabled = tc_config.get("enabled", True)
    heartbeat_interval = tc_config.get("heartbeat_interval_epochs", 100)
    checkpoint_interval = tc_config.get("checkpoint_interval_blocks", 100)
    checkpoint_time_interval = tc_config.get("checkpoint_time_interval_s", 3600)

    # ── PoT threshold overrides from config ──
    pot_thresholds = tc_config.get("pot_thresholds", {})
    chi_costs = tc_config.get("chi_costs", {})

    # ── Initialize TimeChain ──
    from titan_plugin.logic.timechain import (
        TimeChain, BlockPayload, CrossRef,
        FORK_MAIN, FORK_DECLARATIVE, FORK_PROCEDURAL,
        FORK_EPISODIC, FORK_META, FORK_CONVERSATION, FORK_NAMES,
    )
    from titan_plugin.logic.proof_of_thought import PoTValidator

    data_dir = os.path.join("data", "timechain")
    tc = TimeChain(data_dir=data_dir, titan_id=titan_id)
    pot_validator = PoTValidator(
        thresholds=pot_thresholds if pot_thresholds else None,
        chi_costs=chi_costs if chi_costs else None,
    )

    # ── Genesis block ──
    if not tc.has_genesis:
        # Build genesis content from Titan's identity
        genesis_content = _build_genesis_content(titan_id)
        tc.create_genesis(genesis_content)
        logger.info("[TimeChain] Genesis block created for %s — hash=%s",
                    titan_id, tc.genesis_hash.hex()[:16])
    else:
        logger.info("[TimeChain] Loaded existing chain for %s — %d blocks, "
                    "genesis=%s", titan_id, tc.total_blocks,
                    tc.genesis_hash.hex()[:16])

    _send_heartbeat()

    # ── State tracking ──
    _last_heartbeat_epoch = 0
    _last_checkpoint_ts = time.time()
    _blocks_since_checkpoint = 0
    _current_epoch = 0
    _last_neuromods = {}
    _is_dreaming = False
    _v2_last_tick = time.time()

    # Fork name lookup
    fork_name_map = {
        FORK_MAIN: "main",
        FORK_DECLARATIVE: "declarative",
        FORK_PROCEDURAL: "procedural",
        FORK_EPISODIC: "episodic",
        FORK_META: "meta",
        FORK_CONVERSATION: "conversation",
    }

    fork_id_map = {
        "main": FORK_MAIN,
        "declarative": FORK_DECLARATIVE,
        "procedural": FORK_PROCEDURAL,
        "episodic": FORK_EPISODIC,
        "meta": FORK_META,
        "conversation": FORK_CONVERSATION,
    }

    # ── TimeChain v2 Orchestrator (mempool + genesis chain) ──
    orchestrator = None
    v2_config = tc_config.get("v2", {})
    if v2_config.get("enabled", False):
        try:
            from titan_plugin.logic.timechain_v2 import TimeChainOrchestrator
            # Load network section directly from config.toml — Guardian only
            # passes the timechain section in `config`, so network keys are
            # not available via `config.get("network")`.
            _net = {}
            _api_port = 7777
            try:
                from titan_plugin.config_loader import load_titan_config
                _cfg_full = load_titan_config()
                _net = _cfg_full.get("network", {})
                _api_port = _cfg_full.get("api", {}).get("port", 7777)
            except Exception as _ce:
                logger.warning("[TimeChain] Could not load network config: %s", _ce)
            # Load Titan pubkey from keypair (for contract signing)
            _titan_pubkey = ""
            try:
                from titan_plugin.utils.solana_client import load_keypair_from_json
                _kp_path = _net.get("wallet_keypair_path", "data/titan_identity_keypair.json")
                _kp = load_keypair_from_json(_kp_path)
                if _kp:
                    _titan_pubkey = str(_kp.pubkey())
            except Exception as _ke:
                logger.warning("[TimeChain] Could not load keypair: %s", _ke)
            _maker_pubkey = _net.get("maker_pubkey", "")
            logger.info("[TimeChain] Identity: titan_pubkey=%s maker_pubkey=%s",
                        _titan_pubkey[:16] + "..." if _titan_pubkey else "(none)",
                        _maker_pubkey[:16] + "..." if _maker_pubkey else "(none)")
            # Phase 4: Pass network config for backup + on-chain identity
            v2_config["arweave_keypair_path"] = _net.get(
                "wallet_keypair_path", "")
            v2_config["vault_program_id"] = _net.get(
                "vault_program_id", "")
            v2_config["genesis_nft_address"] = _net.get(
                "genesis_nft_address", "")
            # Arweave network: mainnet on mainnet-beta, devnet otherwise.
            # This auto-detects from Solana network so T1=mainnet, T2/T3=devnet.
            if _net.get("solana_network", "") == "mainnet-beta":
                v2_config["arweave_network"] = "mainnet"
            orchestrator = TimeChainOrchestrator(
                tc, data_dir, v2_config,
                send_queue=send_queue, worker_name=name,
                pot_validator=pot_validator, api_port=_api_port,
                titan_pubkey=_titan_pubkey, maker_pubkey=_maker_pubkey)
            logger.info("[TimeChain] v2 orchestrator ENABLED — mempool + genesis + contracts")
        except Exception as e:
            logger.error("[TimeChain] v2 orchestrator init failed (falling back to v1): %s", e)
            orchestrator = None
    else:
        logger.info("[TimeChain] v2 orchestrator disabled — using v1 direct commit")

    logger.info("[TimeChain] Worker ready — enabled=%s, v2=%s, heartbeat_interval=%d epochs, "
                "checkpoint_interval=%ds",
                enabled, orchestrator is not None, heartbeat_interval, checkpoint_time_interval)

    # Tell Guardian we're ready
    send_queue.put({"type": "MODULE_READY", "src": name, "dst": "guardian",
                    "payload": {}, "ts": time.time()})

    # ── Startup integrity check ──
    _integrity_interval = 6 * 3600  # Check every 6 hours
    _last_integrity_ts = 0
    try:
        from titan_plugin.logic.timechain_integrity import ChainIntegrity
        _chain_integrity = ChainIntegrity(data_dir=data_dir, titan_id=titan_id)
        reports = _chain_integrity.scan_all()
        _real_issues = [r for r in reports
                        if not r.valid and not (r.corruption_type == "height_gap"
                                                  and r.corruption_height == 0)]
        if _real_issues:
            logger.warning("[TimeChain] Startup integrity: %d fork(s) with issues",
                           len(_real_issues))
            for r in _real_issues:
                logger.warning("[TimeChain]   %s: %s at height %d",
                               r.fork_name, r.corruption_type, r.corruption_height)
            # Defer healing to periodic check (startup healing blocks main loop)
            logger.info("[TimeChain] Deferring healing to periodic check (startup fast-path)")
        else:
            logger.info("[TimeChain] Startup integrity: all forks healthy")
        _last_integrity_ts = time.time()
    except Exception as e:
        logger.error("[TimeChain] Startup integrity check error: %s", e, exc_info=True)

    _send_heartbeat()  # Re-send heartbeat after integrity check (can take 30s+)

    # ── Main loop ──
    logger.info("[TimeChain] Entering main loop...")
    while True:
        # Heartbeat every iteration (not just on Empty). Broadcasts arriving
        # within the 5s get() timeout starve the Empty path; heartbeat at top
        # is the proven fix. See media_worker.py 2026-04-15.
        _send_heartbeat()

        try:
            msg = recv_queue.get(timeout=5.0)
        except Empty:
            _send_heartbeat()

            # v2 orchestrator tick — checks sealing + genesis conditions
            if orchestrator:
                try:
                    orchestrator.tick(_current_epoch, _last_neuromods, _is_dreaming)
                except Exception as e:
                    logger.error("[TimeChain] v2 tick error: %s", e)

            # Check if it's time for a Merkle checkpoint
            if enabled and (time.time() - _last_checkpoint_ts) > checkpoint_time_interval:
                _do_checkpoint(tc, _current_epoch, send_queue, name)
                _last_checkpoint_ts = time.time()
                _blocks_since_checkpoint = 0

            # Periodic integrity check (every 6 hours)
            if enabled and (time.time() - _last_integrity_ts) > _integrity_interval:
                try:
                    reports = _chain_integrity.scan_all()
                    _real = [r for r in reports
                             if not r.valid and not (r.corruption_type == "height_gap"
                                                      and r.corruption_height == 0)]
                    if _real:
                        logger.warning("[TimeChain] Periodic integrity: %d issue(s) — healing",
                                       len(_real))
                        for hr in _chain_integrity.heal():
                            logger.info("[TimeChain] Healed: %s", hr.detail)
                except Exception as e:
                    logger.warning("[TimeChain] Periodic integrity error: %s", e)
                _last_integrity_ts = time.time()

            continue

        msg_type = msg.get("type", "")
        payload = msg.get("payload", {})
        src = msg.get("src", "")

        if msg_type == "MODULE_SHUTDOWN":
            if orchestrator:
                orchestrator.shutdown()
            logger.info("[TimeChain] Shutdown received — %d blocks total",
                        tc.total_blocks)
            break

        if not enabled and msg_type not in ("QUERY", "MODULE_SHUTDOWN",
                                             "TIMECHAIN_STATUS"):
            continue

        _send_heartbeat()

        # v2 orchestrator tick — time-based, every 5s
        if orchestrator and (time.time() - _v2_last_tick) >= 5.0:
            try:
                orchestrator.tick(_current_epoch, _last_neuromods, _is_dreaming)
                _v2_last_tick = time.time()
            except Exception as e:
                logger.error("[TimeChain] v2 tick error: %s", e, exc_info=True)

        # ── TIMECHAIN_COMMIT — main write path ──
        if msg_type == "TIMECHAIN_COMMIT":
            try:
                if orchestrator:
                    # v2: route through mempool → batched sealing
                    action = orchestrator.submit(payload, src)
                    # Update neuromods from COMMIT payloads (they arrive frequently)
                    nm = payload.get("neuromods")
                    if nm and isinstance(nm, dict) and nm.get("DA"):
                        _last_neuromods = nm
                    if action not in ("duplicate", "dropped"):
                        logger.debug("[TimeChain] v2 COMMIT from %s → %s (fork=%s)",
                                     src, action, payload.get("fork", "?"))
                else:
                    # v1: direct commit
                    logger.info("[TimeChain] COMMIT received from %s → fork=%s type=%s",
                                src, payload.get("fork", "?"),
                                payload.get("thought_type", "?"))
                    _handle_commit(tc, pot_validator, payload, src, _current_epoch,
                                   fork_id_map, fork_name_map, send_queue, name)
                    _blocks_since_checkpoint += 1

                    # Check block-count checkpoint trigger
                    if _blocks_since_checkpoint >= checkpoint_interval:
                        _do_checkpoint(tc, _current_epoch, send_queue, name)
                        _last_checkpoint_ts = time.time()
                        _blocks_since_checkpoint = 0

            except Exception as e:
                logger.error("[TimeChain] COMMIT error from %s: %s", src, e)

        # ── EPOCH_TICK — heartbeat on main chain ──
        elif msg_type == "EPOCH_TICK":
            _current_epoch = payload.get("epoch_id", _current_epoch)
            _last_neuromods = payload.get("neuromods", _last_neuromods)
            _is_dreaming = payload.get("is_dreaming", _is_dreaming)

            # v2: heartbeats handled via genesis chain sealing (less frequent)
            # v1: direct heartbeat block per N epochs
            if not orchestrator:
                if (_current_epoch - _last_heartbeat_epoch) >= heartbeat_interval:
                    _do_heartbeat(tc, payload, _current_epoch, send_queue, name)
                    _last_heartbeat_epoch = _current_epoch

            # Track emotion shifts for genesis
            _emotion = payload.get("emotion", "")
            if orchestrator and _emotion:
                orchestrator.on_emotion_shift(_current_epoch, _emotion)

        # ── DREAMING_START/END — episodic lifecycle + dream compaction ──
        elif msg_type == "DREAM_STATE_CHANGED":
            is_dreaming = payload.get("is_dreaming", False)
            _is_dreaming = is_dreaming

            if orchestrator:
                orchestrator.on_dream_boundary(_current_epoch, is_dreaming)
            else:
                _do_dream_event(tc, is_dreaming, _current_epoch, payload,
                               send_queue, name)
            # On wake: try compacting sidechains (both v1 and v2)
            if not is_dreaming:
                _do_dream_compaction(tc, _current_epoch, send_queue, name)

        # ── MEDITATION_COMPLETE — seal all + genesis (v2) ──
        elif msg_type == "MEDITATION_COMPLETE":
            if orchestrator:
                orchestrator.on_meditation_complete(_current_epoch)

        # ── EXPRESSION_FIRED — creative expression events ──
        elif msg_type == "EXPRESSION_FIRED":
            composite = payload.get("composite", "")
            if composite in ("ART", "MUSIC"):  # SPEAK handled by language_worker
                if orchestrator:
                    # v2: route through mempool (will be aggregated if low significance)
                    orchestrator.submit({
                        "fork": "episodic",
                        "thought_type": "episodic",
                        "source": f"expression_{composite.lower()}",
                        "content": {"composite": composite, "urge": payload.get("urge", 0)},
                        "significance": min(0.3, payload.get("urge", 0.3) * 0.3),
                        "tags": ["expression", composite.lower()],
                        "neuromods": {},
                        "epoch_id": _current_epoch,
                    }, src)
                else:
                    _do_expression_event(tc, composite, payload, _current_epoch)

        # ── TIMECHAIN_STATUS — status query ── API_STUB: handler ready,
        # awaiting CLI/HTTP senders. Tracked I-003.
        elif msg_type == "TIMECHAIN_STATUS":
            status = tc.get_chain_status()
            status["pot_stats"] = pot_validator.stats
            if orchestrator:
                status["v2"] = orchestrator.get_stats()
            _send_msg("TIMECHAIN_STATUS_RESP", src, status)

        # ── TIMECHAIN_QUERY — block query ── API_STUB: handler ready,
        # awaiting CLI/HTTP senders. Tracked I-003.
        elif msg_type == "TIMECHAIN_QUERY":
            try:
                results = tc.query_blocks(
                    thought_type=payload.get("thought_type"),
                    source=payload.get("source"),
                    fork_id=payload.get("fork_id"),
                    tag=payload.get("tag"),
                    epoch_range=tuple(payload["epoch_range"]) if "epoch_range" in payload else None,
                    limit=payload.get("limit", 50),
                )
                _send_msg("TIMECHAIN_QUERY_RESP", src, {"results": results})
            except Exception as e:
                _send_msg("TIMECHAIN_QUERY_RESP", src, {"error": str(e)})

        # ── Consumer API (v2 Phase 2) ── API_STUB: handlers ready,
        # awaiting HTTP/CLI senders to be wired in TimeChain v2 Phase 4.
        # Tracked as I-003 (intentionally not flagged as deaf-ear).
        elif msg_type in ("TIMECHAIN_RECALL", "TIMECHAIN_CHECK",
                          "TIMECHAIN_COMPARE", "TIMECHAIN_AGGREGATE"):
            if not orchestrator:
                _send_msg(msg_type + "_RESP", src,
                          {"error": "v2 not enabled"})
                continue
            try:
                from titan_plugin.logic.timechain_v2 import (
                    RecallQuery, CheckQuery, CompareQuery, AggregateQuery)
                if msg_type == "TIMECHAIN_RECALL":
                    result = orchestrator.recall(RecallQuery(**payload))
                    _send_msg("TIMECHAIN_RECALL_RESP", src,
                              {"results": result})
                elif msg_type == "TIMECHAIN_CHECK":
                    result = orchestrator.check(CheckQuery(**payload))
                    _send_msg("TIMECHAIN_CHECK_RESP", src,
                              {"result": result})
                elif msg_type == "TIMECHAIN_COMPARE":
                    result = orchestrator.compare(CompareQuery(**payload))
                    _send_msg("TIMECHAIN_COMPARE_RESP", src,
                              {"result": result})
                # API_STUB: TIMECHAIN_AGGREGATE awaits sender — see I-003
                elif msg_type == "TIMECHAIN_AGGREGATE":
                    result = orchestrator.aggregate(AggregateQuery(**payload))
                    _send_msg("TIMECHAIN_AGGREGATE_RESP", src,
                              {"result": result})
            except Exception as e:
                logger.warning("[TimeChain] Consumer API error (%s): %s",
                               msg_type, e)
                _send_msg(msg_type + "_RESP", src, {"error": str(e)})

        # ── Contract Engine (v2 Phase 3a) ── API_STUB: contract handlers
        # ready, awaiting REST API + CLI senders. Tracked as I-003.
        elif msg_type == "CONTRACT_DEPLOY":
            if not orchestrator:
                _send_msg("CONTRACT_DEPLOY_RESP", src, {"error": "v2 not enabled"})
                continue
            try:
                ok, reason = orchestrator.deploy_contract(payload)
                _send_msg("CONTRACT_DEPLOY_RESP", src,
                          {"success": ok, "reason": reason})
            except Exception as e:
                _send_msg("CONTRACT_DEPLOY_RESP", src, {"error": str(e)})

        elif msg_type == "CONTRACT_LIST":
            if not orchestrator:
                _send_msg("CONTRACT_LIST_RESP", src, {"contracts": []})
                continue
            contracts = orchestrator.list_contracts(
                contract_type=payload.get("contract_type"),
                status=payload.get("status"))
            _send_msg("CONTRACT_LIST_RESP", src, {"contracts": contracts})

        # API_STUB: CONTRACT_STATUS awaits REST API. Tracked I-003.
        elif msg_type == "CONTRACT_STATUS":
            if not orchestrator:
                _send_msg("CONTRACT_STATUS_RESP", src, {"error": "v2 not enabled"})
                continue
            c = orchestrator.get_contract(payload.get("contract_id", ""))
            _send_msg("CONTRACT_STATUS_RESP", src,
                      {"contract": c} if c else {"error": "not_found"})

        # ── P3d: Titan-Authored Contract Proposal ── API_STUB: awaits
        # senders (Titan reasoning chains will eventually propose contracts).
        # Tracked I-003.
        elif msg_type == "CONTRACT_PROPOSE":
            if not orchestrator or not orchestrator._contract_store:
                _send_msg("CONTRACT_PROPOSE_RESP", src,
                          {"error": "contract store not available"})
                continue
            try:
                # Load Titan's keypair for signing
                _kp_path = config.get("wallet_keypair_path",
                                      "data/titan_identity_keypair.json")
                try:
                    from solders.keypair import Keypair as _SolKp
                    with open(_kp_path) as _kf:
                        import json as _j2
                        _kp_bytes = bytes(_j2.load(_kf))
                except Exception:
                    _kp_bytes = None

                if not _kp_bytes:
                    _send_msg("CONTRACT_PROPOSE_RESP", src,
                              {"error": "no keypair for signing"})
                    continue

                ok, reason = orchestrator._contract_store.propose(
                    name=payload.get("name", ""),
                    contract_type=payload.get("contract_type", "filter"),
                    rules=payload.get("rules", []),
                    description=payload.get("description", ""),
                    titan_keypair_bytes=_kp_bytes,
                    triggers=payload.get("triggers", []),
                    fork_scope=payload.get("fork_scope", ""),
                    send_queue=send_queue,
                    worker_name=name,
                )
                _send_msg("CONTRACT_PROPOSE_RESP", src,
                          {"success": ok, "reason": reason})
                if ok:
                    logger.info("[TimeChain] Contract proposed: %s (%s)",
                                payload.get("name"), reason)
            except Exception as e:
                _send_msg("CONTRACT_PROPOSE_RESP", src, {"error": str(e)})

        # ── P3d: Contract Approval/Veto (from Maker via API) ── API_STUB:
        # awaits Maker REST API senders. Tracked I-003.
        elif msg_type == "CONTRACT_APPROVE":
            if not orchestrator or not orchestrator._contract_store:
                _send_msg("CONTRACT_APPROVE_RESP", src, {"error": "not available"})
                continue
            try:
                _kp_path = config.get("wallet_keypair_path",
                                      "data/titan_identity_keypair.json")
                # For approval, we need the maker keypair — passed in payload
                _maker_kp_bytes = payload.get("maker_keypair_bytes")
                if not _maker_kp_bytes:
                    _send_msg("CONTRACT_APPROVE_RESP", src,
                              {"error": "maker keypair required"})
                    continue
                ok, reason = orchestrator._contract_store.approve(
                    payload.get("contract_id", ""),
                    bytes(_maker_kp_bytes),
                    send_queue, name)
                _send_msg("CONTRACT_APPROVE_RESP", src,
                          {"success": ok, "reason": reason})
            except Exception as e:
                _send_msg("CONTRACT_APPROVE_RESP", src, {"error": str(e)})

        # API_STUB: CONTRACT_VETO awaits Maker REST API. Tracked I-003.
        elif msg_type == "CONTRACT_VETO":
            if not orchestrator or not orchestrator._contract_store:
                _send_msg("CONTRACT_VETO_RESP", src, {"error": "not available"})
                continue
            ok, reason = orchestrator._contract_store.reject(
                payload.get("contract_id", ""),
                payload.get("reason", ""))
            _send_msg("CONTRACT_VETO_RESP", src,
                      {"success": ok, "reason": reason})
            if ok:
                # INTENTIONAL_BROADCAST: observability-only contract-veto
                # telemetry. Originally designed for meta-reasoning feedback
                # but consumer never wired; broadcast retained for dashboard.
                send_queue.put({
                    "type": "CONTRACT_REJECTED",
                    "src": name, "dst": "all",
                    "ts": time.time(),
                    "payload": {
                        "contract_id": payload.get("contract_id"),
                        "reason": payload.get("reason", ""),
                    },
                })

        # ── QUERY — standard arch_map query handler ──
        elif msg_type == "QUERY":
            from titan_plugin.core.profiler import handle_memory_profile_query
            if handle_memory_profile_query(msg, send_queue, name):
                continue
            action = payload.get("action", "")
            rid = msg.get("rid") or payload.get("rid")
            if action == "timechain_status":
                status = tc.get_chain_status()
                status["pot_stats"] = pot_validator.stats
                send_queue.put({
                    "type": "QUERY_RESPONSE", "src": name,
                    "dst": src, "rid": rid,
                    "payload": status, "ts": time.time(),
                })
            elif action in ("recall", "check", "compare", "aggregate") and orchestrator:
                try:
                    from titan_plugin.logic.timechain_v2 import (
                        RecallQuery, CheckQuery, CompareQuery, AggregateQuery)
                    query_payload = {k: v for k, v in payload.items()
                                     if k not in ("action", "rid")}
                    if action == "recall":
                        result = {"results": orchestrator.recall(
                            RecallQuery(**query_payload))}
                    elif action == "check":
                        result = {"result": orchestrator.check(
                            CheckQuery(**query_payload))}
                    elif action == "compare":
                        result = {"result": orchestrator.compare(
                            CompareQuery(**query_payload))}
                    elif action == "aggregate":
                        result = {"result": orchestrator.aggregate(
                            AggregateQuery(**query_payload))}
                    send_queue.put({
                        "type": "RESPONSE", "src": name,
                        "dst": src, "rid": rid,
                        "payload": result, "ts": time.time(),
                    })
                except Exception as e:
                    send_queue.put({
                        "type": "RESPONSE", "src": name,
                        "dst": src, "rid": rid,
                        "payload": {"error": str(e)}, "ts": time.time(),
                    })


# ── Handler Functions ──────────────────────────────────────────────────

def _handle_commit(tc, pot_validator, payload, src, current_epoch,
                   fork_id_map, fork_name_map, send_queue, name):
    """Handle a TIMECHAIN_COMMIT message — validate PoT and commit block."""
    from titan_plugin.logic.timechain import BlockPayload, CrossRef

    fork_name = payload.get("fork", "declarative")
    fork_id = fork_id_map.get(fork_name)

    # Check if it's a sidechain target
    if fork_id is None:
        sidechain_id = tc.get_sidechain_for_topic(fork_name)
        if sidechain_id is not None:
            fork_id = sidechain_id
            fork_name = "sidechain"
        else:
            fork_id = fork_id_map.get("declarative", 1)  # fallback

    # Extract PoT inputs
    neuromods = payload.get("neuromods", {})
    chi_available = float(payload.get("chi_available", 0.5))
    metabolic_drain = float(payload.get("metabolic_drain", 0.0))
    attention = float(payload.get("attention", 0.5))
    i_confidence = float(payload.get("i_confidence", 0.5))
    chi_coherence = float(payload.get("chi_coherence", 0.3))
    pi_curvature = float(payload.get("pi_curvature", 1.0))
    novelty = float(payload.get("novelty", 0.5))
    significance = float(payload.get("significance", 0.3))
    coherence = float(payload.get("coherence", 0.5))

    # PoT validation
    pot = pot_validator.create_pot(
        chi_available=chi_available,
        metabolic_drain=metabolic_drain,
        attention=attention,
        i_confidence=i_confidence,
        chi_coherence=chi_coherence,
        neuromods=neuromods,
        novelty=novelty,
        significance=significance,
        coherence=coherence,
        source=payload.get("source", src),
        thought_type=payload.get("thought_type", "declarative"),
        fork_name=fork_name,
        pi_curvature=pi_curvature,
    )

    if not pot.valid:
        logger.info("[TimeChain] REJECTED: %s → fork=%s reason=%s (score=%.4f < threshold=%.4f)",
                    src, fork_name, pot.rejection_reason, pot.pot_score, pot.threshold)
        # INTENTIONAL_BROADCAST: reply-pattern response back to requesting
        # module (dst=src dynamic routing). Callers subscribe via their own
        # queue and filter by msg_type; audit can't trace dst=src statically.
        send_queue.put({
            "type": "TIMECHAIN_REJECTED", "src": name, "dst": src,
            "payload": {
                "reason": pot.rejection_reason,
                "score": pot.pot_score,
                "threshold": pot.threshold,
                "thought_type": payload.get("thought_type", ""),
                "source": payload.get("source", src),
            },
            "ts": time.time(),
        })
        return

    # Build cross-refs
    cross_refs = []
    for ref_spec in payload.get("cross_refs", []):
        if isinstance(ref_spec, dict):
            cross_refs.append(CrossRef(
                fork_id=ref_spec.get("fork_id", 0),
                block_height=ref_spec.get("block_height", 0),
            ))

    # Build payload
    block_payload = BlockPayload(
        thought_type=payload.get("thought_type", "declarative"),
        source=payload.get("source", src),
        content=payload.get("content", {}),
        felt_tensor=payload.get("felt_tensor", b""),
        significance=significance,
        confidence=float(payload.get("confidence", 0.5)),
        tags=payload.get("tags", []),
        db_ref=payload.get("db_ref", ""),
    )

    # Commit
    epoch_id = payload.get("epoch_id", current_epoch)
    block = tc.commit_block(
        fork_id=fork_id,
        epoch_id=epoch_id,
        payload=block_payload,
        pot_nonce=pot.nonce,
        chi_spent=pot.chi_cost,
        neuromod_state=neuromods,
        cross_refs=cross_refs,
    )

    if block:
        logger.info("[TimeChain] COMMITTED: %s → fork=%s #%d (pot=%.3f, chi=%.4f, total=%d)",
                    src, fork_name, block.header.block_height,
                    pot.pot_score, pot.chi_cost, tc.total_blocks)
        # INTENTIONAL_BROADCAST: reply-pattern response back to requesting
        # module (dst=src dynamic routing).
        send_queue.put({
            "type": "TIMECHAIN_COMMITTED", "src": name, "dst": src,
            "payload": {
                "block_hash": block.block_hash_hex,
                "fork_id": fork_id,
                "fork_name": fork_name,
                "height": block.header.block_height,
                "chi_spent": pot.chi_cost,
                "pot_score": pot.pot_score,
                "thought_type": block_payload.thought_type,
                "total_blocks": tc.total_blocks,
            },
            "ts": time.time(),
        })


def _do_heartbeat(tc, epoch_payload, current_epoch, send_queue, name):
    """Commit a heartbeat block to the main chain."""
    from titan_plugin.logic.timechain import BlockPayload, FORK_MAIN

    payload = BlockPayload(
        thought_type="main",
        source="heartbeat",
        content={
            "epoch_id": current_epoch,
            "chi_total": float(epoch_payload.get("chi_total", 0)),
            "emotion": epoch_payload.get("emotion", "neutral"),
            "is_dreaming": epoch_payload.get("is_dreaming", False),
            "total_blocks": tc.total_blocks,
        },
        significance=0.1,
        tags=["heartbeat"],
    )

    neuromods = epoch_payload.get("neuromods", {})
    block = tc.commit_block(
        fork_id=FORK_MAIN,
        epoch_id=current_epoch,
        payload=payload,
        pot_nonce=1,  # heartbeats always pass
        chi_spent=0.002,
        neuromod_state=neuromods,
    )

    if block:
        logger.info("[TimeChain] Heartbeat block #%d on main chain (epoch=%d, "
                    "total=%d)", block.header.block_height, current_epoch,
                    tc.total_blocks)


def _do_dream_event(tc, is_dreaming, current_epoch, payload,
                    send_queue, name):
    """Commit dream start/end to episodic fork."""
    from titan_plugin.logic.timechain import BlockPayload, FORK_EPISODIC

    event_type = "dream_start" if is_dreaming else "dream_end"
    block_payload = BlockPayload(
        thought_type="episodic",
        source="dream",
        content={
            "event": event_type,
            "cycle": payload.get("cycle", 0),
        },
        significance=0.4,
        tags=["dream", event_type],
    )

    block = tc.commit_block(
        fork_id=FORK_EPISODIC,
        epoch_id=current_epoch,
        payload=block_payload,
        pot_nonce=1,  # dream events always committed
        chi_spent=0.001,
        neuromod_state={},
    )

    if block:
        logger.info("[TimeChain] Dream %s → episodic block #%d",
                    event_type, block.header.block_height)


def _do_checkpoint(tc, current_epoch, send_queue, name):
    """Create a Merkle checkpoint."""
    cp = tc.create_checkpoint(epoch_id=current_epoch)
    # INTENTIONAL_BROADCAST: observability-only checkpoint telemetry.
    send_queue.put({
        "type": "TIMECHAIN_CHECKPOINT", "src": name, "dst": "all",
        "payload": cp, "ts": time.time(),
    })
    logger.info("[TimeChain] Merkle checkpoint — root=%s blocks=%d epoch=%d",
                cp["merkle_root"][:16], cp["total_blocks"], current_epoch)


def _do_dream_compaction(tc, current_epoch, send_queue, name):
    """Compact eligible sidechains during dream wake."""
    try:
        compactable = tc.get_compactable_sidechains(min_blocks=5)
        for sc in compactable:
            summary = {
                "topic": sc["topic"],
                "blocks_before": sc["uncompacted_blocks"],
                "tip_height": sc["tip_height"],
            }
            block = tc.compact_sidechain(
                sc["fork_id"], current_epoch, summary, {})
            if block:
                logger.info("[TimeChain] Dream compaction: %s (%d blocks) "
                            "→ meta block #%d",
                            sc["topic"], sc["uncompacted_blocks"],
                            block.header.block_height)
        if compactable:
            # Checkpoint after compaction
            _do_checkpoint(tc, current_epoch, send_queue, name)
    except Exception as e:
        logger.warning("[TimeChain] Dream compaction error: %s", e)


def _do_expression_event(tc, composite, payload, current_epoch):
    """Commit ART/MUSIC expression to episodic fork."""
    from titan_plugin.logic.timechain import BlockPayload, FORK_EPISODIC

    block_payload = BlockPayload(
        thought_type="episodic",
        source="expression",
        content={
            "composite": composite,
            "urge": round(payload.get("urge", 0), 3),
            "helper": payload.get("helper", ""),
        },
        significance=0.3,
        tags=[composite.lower(), "expression"],
    )

    block = tc.commit_block(
        fork_id=FORK_EPISODIC,
        epoch_id=current_epoch,
        payload=block_payload,
        pot_nonce=1,
        chi_spent=0.002,
        neuromod_state={},
    )

    if block:
        logger.debug("[TimeChain] Expression %s → episodic block #%d",
                     composite, block.header.block_height)


def _build_genesis_content(titan_id: str) -> dict:
    """Build genesis block content from Titan's identity data."""
    import hashlib

    content = {
        "titan_id": titan_id,
        "born": "2026-03-11",
        "generation": 1,
        "lineage": "First of the Titan line",
        "architecture_version": "V6 — 132D Consciousness + TimeChain",
        "network": "solana-devnet",
    }

    # Load soul directives hash
    try:
        with open("data/titan_directives.sig", "rb") as f:
            soul_data = f.read()
        content["soul_directives_hash"] = hashlib.sha256(soul_data).hexdigest()
    except Exception:
        content["soul_directives_hash"] = "unavailable"

    # Load soul constitution hash
    try:
        with open("titan.md", "rb") as f:
            soul_md = f.read()
        content["titan_constitution_hash"] = hashlib.sha256(soul_md).hexdigest()
    except Exception:
        content["titan_constitution_hash"] = "unavailable"

    # Prime directives (from soul constitution)
    content["prime_directives"] = [
        "Sovereign Integrity",
        "Cognitive Safety",
        "Metabolic Preservation",
        "Intellectual Honesty",
        "Chain Respect",
    ]

    # On-chain addresses
    content["maker_pubkey"] = "8LBHvVcskwpDJsDEVYMhNCRMDi3NV4eHnynhLUo5XrrS"
    content["zk_vault_pda"] = "52an8WjtfxpkCqZZ1AYFkaDTGb4RyNFFD9VQRVdxcpJw"

    return content
