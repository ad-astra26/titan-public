//! broker — Top-level `BusBroker` that wires accept loop + per-connection
//! tasks + heartbeat + slow-consumer + fanout routing.
//!
//! Byte-identical port of Python `BusSocketServer` start/stop/publish API.
//!
//! # Lifecycle
//!
//! ```ignore
//! let broker = BusBroker::new("T1", authkey_bytes);
//! broker.start("/tmp/titan_bus_T1.sock").await?;
//! // ... broker runs in background tasks ...
//! broker.stop().await;
//! ```
//!
//! # Routing rules (mirror Python `BusSocketServer.publish`)
//!
//! - `dst == "all"` or absent: deliver to every subscriber (skip publisher
//!   itself to avoid echo loops)
//! - Specific `dst`: deliver only to the subscriber whose name matches
//!
//! Drift bridges (D13/D14/D15) dual-emit per
//! [`crate::drift_bridge::bridge_emit_names`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::net::UnixListener;
use tokio::sync::{mpsc, Mutex, Notify};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use crate::boot_buffer::{BootBuffer, BootBufferPushOutcome};
use crate::drift_bridge::bridge_emit_names;
use crate::heartbeat::run_heartbeat_loop;
use crate::message::{rewrite_msg_type, MsgHeader};
use crate::ring::Envelope;
use crate::server::{perform_handshake, run_recv_loop, InboundEvent};
use crate::slow_consumer::run_slow_consumer_loop;
use crate::subscriber::{BrokerSubscriber, SubscriberMap};
use titan_core::bus_specs::{get_spec, Priority};

/// Errors during broker lifecycle.
#[derive(Debug, thiserror::Error)]
pub enum BrokerError {
    /// Broker not in a state to perform the requested operation.
    #[error("broker not running (or already stopped)")]
    NotRunning,
    /// Failed to bind the Unix socket.
    #[error("failed to bind {path}: {source}")]
    BindFailed {
        /// Socket path attempted.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// I/O error during operation.
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}

/// Top-level broker handle. One per Titan kernel.
pub struct BusBroker {
    titan_id: String,
    authkey: Arc<Vec<u8>>,
    subs: Arc<Mutex<SubscriberMap>>,
    notify_per_sub: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
    inbound_tx: Option<mpsc::UnboundedSender<(String, InboundEvent)>>,
    accept_handle: Option<JoinHandle<()>>,
    inbound_handle: Option<JoinHandle<()>>,
    heartbeat_handle: Option<JoinHandle<()>>,
    slow_consumer_handle: Option<JoinHandle<()>>,
    shutdown: Arc<Notify>,
    sock_path: Option<PathBuf>,
    anon_counter: Arc<Mutex<u64>>,
    /// SPEC §8.0.bis boot-window buffer — holds targeted P0 messages
    /// whose destination subscriber has not yet attached. Drained on
    /// BUS_SUBSCRIBE; TTL-evicted on lazy GC during fanout.
    boot_buffer: Arc<Mutex<BootBuffer>>,
}

impl BusBroker {
    /// Construct a new broker (does NOT bind the socket — call [`start`]).
    ///
    /// `titan_id` is currently informational; future use for per-Titan
    /// observability tagging.
    pub fn new(titan_id: impl Into<String>, authkey: Vec<u8>) -> Self {
        Self {
            titan_id: titan_id.into(),
            authkey: Arc::new(authkey),
            subs: Arc::new(Mutex::new(HashMap::new())),
            notify_per_sub: Arc::new(Mutex::new(HashMap::new())),
            inbound_tx: None,
            accept_handle: None,
            inbound_handle: None,
            heartbeat_handle: None,
            slow_consumer_handle: None,
            shutdown: Arc::new(Notify::new()),
            sock_path: None,
            anon_counter: Arc::new(Mutex::new(0)),
            boot_buffer: Arc::new(Mutex::new(BootBuffer::new())),
        }
    }

    /// Bind the Unix socket + start accept + heartbeat + slow-consumer loops.
    pub async fn start(&mut self, sock_path: impl Into<PathBuf>) -> Result<(), BrokerError> {
        let sock_path = sock_path.into();

        // Idempotent-failsafe: unlink stale socket file
        if sock_path.exists() {
            let _ = std::fs::remove_file(&sock_path);
        }

        let listener = UnixListener::bind(&sock_path).map_err(|e| BrokerError::BindFailed {
            path: sock_path.clone(),
            source: e,
        })?;

        // Set socket file mode 0600 (owner-only) — same as Python broker
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&sock_path, std::fs::Permissions::from_mode(0o600));
        }

        info!(titan_id = %self.titan_id, path = ?sock_path, "bus broker listening");

        // Inbound channel: recv tasks → broker dispatcher
        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
        self.inbound_tx = Some(inbound_tx.clone());

        // Spawn dispatcher task
        self.inbound_handle = Some(tokio::spawn(Self::run_inbound_dispatcher(
            inbound_rx,
            self.subs.clone(),
            self.notify_per_sub.clone(),
            self.boot_buffer.clone(),
        )));

        // Spawn heartbeat loop
        self.heartbeat_handle = Some(tokio::spawn(run_heartbeat_loop(
            self.subs.clone(),
            self.notify_per_sub.clone(),
            self.shutdown.clone(),
        )));

        // Spawn slow-consumer loop
        self.slow_consumer_handle = Some(tokio::spawn(run_slow_consumer_loop(
            self.subs.clone(),
            self.shutdown.clone(),
        )));

        // Spawn accept loop
        self.accept_handle = Some(tokio::spawn(Self::run_accept_loop(
            listener,
            self.authkey.clone(),
            self.subs.clone(),
            self.notify_per_sub.clone(),
            inbound_tx,
            self.anon_counter.clone(),
            self.shutdown.clone(),
        )));

        self.sock_path = Some(sock_path);
        Ok(())
    }

    /// Graceful shutdown: signal all tasks, await joins, unlink socket.
    pub async fn stop(&mut self) {
        // Notify all background loops
        for _ in 0..4 {
            // Heartbeat + slow_consumer + accept loop both wait on the same
            // shutdown notify. notify_waiters() wakes ALL pending listeners.
            self.shutdown.notify_waiters();
        }

        // Mark all subscribers closed so send tasks exit
        {
            let mut subs = self.subs.lock().await;
            for sub in subs.values_mut() {
                sub.closed = true;
            }
        }
        // Wake all send tasks so they observe the closed flag and exit
        for notify in self.notify_per_sub.lock().await.values() {
            notify.notify_waiters();
        }

        // Drop inbound channel — dispatcher exits when channel closes
        self.inbound_tx = None;

        // Wait for tasks
        if let Some(h) = self.accept_handle.take() {
            h.abort();
            let _ = h.await;
        }
        if let Some(h) = self.inbound_handle.take() {
            h.abort();
            let _ = h.await;
        }
        if let Some(h) = self.heartbeat_handle.take() {
            h.abort();
            let _ = h.await;
        }
        if let Some(h) = self.slow_consumer_handle.take() {
            h.abort();
            let _ = h.await;
        }

        // Unlink socket file
        if let Some(path) = self.sock_path.take() {
            let _ = std::fs::remove_file(path);
        }

        info!(titan_id = %self.titan_id, "bus broker stopped");
    }

    /// Returns the number of currently-connected subscribers (test helper).
    pub async fn subscriber_count(&self) -> usize {
        self.subs.lock().await.len()
    }

    /// Publish a message from in-process (no Unix socket round-trip).
    ///
    /// Used by the kernel's internal publishers (clocks emitting
    /// `KERNEL_EPOCH_TICK`, supervisor emitting `SUPERVISION_*`). Mirrors
    /// the Python broker's `_on_inbound_publish` callback that lets
    /// kernel-side code fanout to socket subscribers.
    ///
    /// `from_name` is used to skip echo-to-self (kernel's own subscribers
    /// don't see their own publishes). Use a unique label per kernel-internal
    /// publisher (e.g. `"kernel:clocks"`, `"kernel:supervisor"`).
    pub async fn publish_local(&self, msg_type: &str, src: &str, raw_bytes: Vec<u8>) {
        let header = crate::message::MsgHeader {
            msg_type: Some(msg_type.to_string()),
            src: Some(src.to_string()),
            dst: None, // None = "all"
        };
        Self::fanout(
            &self.subs,
            &self.notify_per_sub,
            &self.boot_buffer,
            &format!("kernel:internal:{src}"),
            header,
            raw_bytes,
        )
        .await;
    }

    // ── Internal task loops ─────────────────────────────────────────────

    async fn next_anon_name(counter: &Arc<Mutex<u64>>) -> String {
        let mut c = counter.lock().await;
        *c += 1;
        format!("anon-{}", *c)
    }

    async fn run_accept_loop(
        listener: UnixListener,
        authkey: Arc<Vec<u8>>,
        subs: Arc<Mutex<SubscriberMap>>,
        notify_per_sub: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
        inbound_tx: mpsc::UnboundedSender<(String, InboundEvent)>,
        anon_counter: Arc<Mutex<u64>>,
        shutdown: Arc<Notify>,
    ) {
        loop {
            tokio::select! {
                accept = listener.accept() => {
                    match accept {
                        Ok((stream, _)) => {
                            let authkey = authkey.clone();
                            let subs = subs.clone();
                            let notify_per_sub = notify_per_sub.clone();
                            let inbound_tx = inbound_tx.clone();
                            let name = Self::next_anon_name(&anon_counter).await;
                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_connection(
                                    stream, name.clone(), authkey, subs, notify_per_sub, inbound_tx,
                                ).await {
                                    debug!(name = %name, err = ?e, "connection ended");
                                }
                            });
                        }
                        Err(e) => {
                            error!(err = ?e, "accept failed");
                            return;
                        }
                    }
                }
                _ = shutdown.notified() => {
                    debug!("accept loop: shutdown received");
                    return;
                }
            }
        }
    }

    async fn handle_connection(
        mut stream: tokio::net::UnixStream,
        sub_name: String,
        authkey: Arc<Vec<u8>>,
        subs: Arc<Mutex<SubscriberMap>>,
        notify_per_sub: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
        inbound_tx: mpsc::UnboundedSender<(String, InboundEvent)>,
    ) -> Result<(), BrokerError> {
        // 1. Handshake — capture peer credentials BEFORE handshake for diagnostics
        // so we can identify the failing client process when HMAC mismatch occurs.
        let peer_pid: i64 = stream
            .peer_cred()
            .ok()
            .and_then(|c| c.pid())
            .map(|p| p as i64)
            .unwrap_or(-1);
        if let Err(e) = perform_handshake(&mut stream, &authkey).await {
            warn!(name = %sub_name, peer_pid = peer_pid, err = ?e, "handshake failed; closing");
            return Err(BrokerError::Io(std::io::Error::other(format!("{e:?}"))));
        }

        // 2. Register subscriber + create per-sub notify
        let notify = Arc::new(Notify::new());
        {
            let mut subs_guard = subs.lock().await;
            subs_guard.insert(sub_name.clone(), BrokerSubscriber::new(&sub_name));
            notify_per_sub
                .lock()
                .await
                .insert(sub_name.clone(), notify.clone());
        }

        // 3. Split stream + spawn recv + send tasks concurrently
        let (read_half, write_half) = stream.into_split();
        let sub_arc = {
            // Wrap in Arc<Mutex<>> for the send task
            // The map already owns the BrokerSubscriber; create a parallel
            // ref via Arc<Mutex<>>. Since Rust's borrow checker disallows
            // two-owner mutation, we use a helper SubArc pattern in tests.
            // For production: send loop reads via the shared subs map.
            //
            // Simpler: pass `subs` map + name to send loop; send loop locks
            // and pops from there directly. This matches Python's design
            // (BrokerSubscriber stays in the broker's dict; send_thread reads
            // via the dict).
            SubByName::new(subs.clone(), sub_name.clone())
        };

        let recv_task = tokio::spawn(run_recv_loop(read_half, sub_name.clone(), inbound_tx));
        let send_task = tokio::spawn(run_send_loop_via_map(
            write_half,
            sub_arc.subs,
            sub_arc.name,
            notify,
        ));

        let _ = tokio::join!(recv_task, send_task);

        // 4. Connection over: purge subscriber + notify
        let mut subs_guard = subs.lock().await;
        subs_guard.remove(&sub_name);
        notify_per_sub.lock().await.remove(&sub_name);
        Ok(())
    }

    async fn run_inbound_dispatcher(
        mut inbound_rx: mpsc::UnboundedReceiver<(String, InboundEvent)>,
        subs: Arc<Mutex<SubscriberMap>>,
        notify_per_sub: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
        boot_buffer: Arc<Mutex<BootBuffer>>,
    ) {
        while let Some((sub_name, event)) = inbound_rx.recv().await {
            match event {
                InboundEvent::Subscribe {
                    name,
                    topics,
                    reply_only,
                } => {
                    // Update the subscriber's `name` field in place; map key
                    // remains stable (anon-N). Fanout filters by `sub.name`,
                    // not by map key, so promotion is just a field update.
                    // Avoids invalidating the per-connection notify + send-loop
                    // map-key references.
                    //
                    // 2026-05-12 — Multi-name semantics: the first
                    // BUS_SUBSCRIBE with a non-anon name sets sub.name
                    // (replacing the initial "anon-N"). Subsequent
                    // BUS_SUBSCRIBE frames over the SAME connection with a
                    // DIFFERENT non-anon name are treated as ALIAS
                    // additions (sub.aliases.insert(name)) rather than
                    // primary-name replacement. Re-sends of the same name
                    // are idempotent. Fanout matches dst against both
                    // sub.name and sub.aliases (see fanout() filter
                    // below). Closes BUG-PHASE-C-BUS-FANOUT-MULTI-NAME-
                    // 20260512: kernel-side proxy reply queues
                    // (output_verifier_proxy, agency_proxy, …) need
                    // RESPONSE messages routed to titan_HCL's single
                    // connection without spawning a separate broker
                    // connection per proxy.
                    // Capture the name(s) this connection registers so we
                    // can drain the SPEC §8.0.bis boot-buffer for those
                    // identities after releasing the subs lock. Buffer
                    // drain MUST NOT hold the subs lock — drained frames
                    // are enqueued onto the subscriber's send queue
                    // which itself coordinates with notify_per_sub.
                    let mut drain_names: Vec<String> = Vec::new();
                    {
                        let mut subs_guard = subs.lock().await;
                        if let Some(sub) = subs_guard.get_mut(&sub_name) {
                            if !name.is_empty() {
                                let already_primary = sub.name == name;
                                if sub.name.starts_with("anon-") || already_primary {
                                    sub.name = name.clone();
                                } else {
                                    sub.aliases.insert(name.clone());
                                }
                                // Drain buffer for the registered identity.
                                // Re-sends of the same name are idempotent
                                // for the buffer too (drain of empty queue).
                                drain_names.push(name.clone());
                            }
                            sub.subscribed_topics.extend(topics);
                            // D-SPEC-42 (SPEC v1.4.0, 2026-05-12) — connection-
                            // level subscriber intent. Last value wins on
                            // multi-name subscribe per SPEC §8.2 v1.4.0.
                            // Mirrors Python `BusSocketServer._handle_inbound`
                            // BUS_SUBSCRIBE handler.
                            sub.reply_only = reply_only;
                            // Mark intent declared so broadcast fanout can
                            // distinguish a pre-subscribe transient (silent
                            // skip) from the §8.2 v1.4.0 empty-topics
                            // forbidden regression (loud drop).
                            sub.has_subscribed = true;
                        }
                    }
                    // SPEC §8.0.bis boot-buffer drain — outside subs lock.
                    // For each newly-registered name (primary or alias),
                    // deliver any buffered frames in arrival order.
                    if !drain_names.is_empty() {
                        let now = Instant::now();
                        for dst_name in &drain_names {
                            let drained = {
                                let mut bb = boot_buffer.lock().await;
                                bb.drain(dst_name, now)
                            };
                            if drained.is_empty() {
                                continue;
                            }
                            // Re-deliver each buffered frame as a Publish
                            // event into the inbound dispatch path. This
                            // reuses the canonical fanout — including
                            // closed-sub silent-skip, reply_only checks,
                            // drift-bridge expansion, etc. — without
                            // duplicating those rules here.
                            info!(
                                dst = %dst_name,
                                count = drained.len(),
                                "[boot_buffer] drained {} frames to subscriber {}",
                                drained.len(),
                                dst_name
                            );
                            for frame in drained {
                                // Synthesize a minimal MsgHeader from the
                                // buffered msg_type + targeted dst. The
                                // original raw_bytes are passed through.
                                let header = MsgHeader {
                                    msg_type: Some(frame.msg_type.clone()),
                                    src: None,
                                    dst: Some(dst_name.clone()),
                                };
                                Self::fanout(
                                    &subs,
                                    &notify_per_sub,
                                    &boot_buffer,
                                    "boot_buffer", // virtual src (no echo loop possible)
                                    header,
                                    frame.raw_bytes,
                                )
                                .await;
                            }
                        }
                    }
                }
                InboundEvent::Unsubscribe { topics } => {
                    let mut subs_guard = subs.lock().await;
                    if let Some(sub) = subs_guard.get_mut(&sub_name) {
                        for t in topics {
                            sub.subscribed_topics.remove(&t);
                        }
                    }
                }
                InboundEvent::Pong => {
                    let mut subs_guard = subs.lock().await;
                    if let Some(sub) = subs_guard.get_mut(&sub_name) {
                        sub.note_pong();
                    }
                }
                InboundEvent::Publish { header, raw_bytes } => {
                    Self::fanout(
                        &subs,
                        &notify_per_sub,
                        &boot_buffer,
                        &sub_name,
                        header,
                        raw_bytes,
                    )
                    .await;
                }
            }
        }
        debug!("inbound dispatcher: channel closed");
    }

    async fn fanout(
        subs: &Arc<Mutex<SubscriberMap>>,
        notify_per_sub: &Arc<Mutex<HashMap<String, Arc<Notify>>>>,
        boot_buffer: &Arc<Mutex<BootBuffer>>,
        from_name: &str,
        header: MsgHeader,
        raw_bytes: Vec<u8>,
    ) {
        let dst = header.dst.as_deref().unwrap_or("all");
        let msg_type = header.msg_type.clone().unwrap_or_default();
        let src = header.src.clone().unwrap_or_default();

        // SPEC §8.0.bis boot-window buffer:
        //   if dst != "all" AND no subscriber registered under dst
        //   AND msg_type is boot-buffer-eligible → enqueue to buffer
        //   instead of dropping. Drained on next BUS_SUBSCRIBE that
        //   registers dst as primary name or alias.
        // Also performs lazy TTL GC on every fanout call (cheap; typical
        // buffer state is empty in steady state).
        if dst != "all" {
            let now = Instant::now();
            let has_subscriber = {
                let subs_guard = subs.lock().await;
                subs_guard
                    .values()
                    .any(|sub| !sub.closed && (sub.name == dst || sub.aliases.contains(dst)))
            };
            if !has_subscriber {
                let outcome = {
                    let mut bb = boot_buffer.lock().await;
                    bb.gc(now); // lazy GC every fanout call
                    bb.push(dst, msg_type.clone(), raw_bytes.clone(), now)
                };
                match outcome {
                    BootBufferPushOutcome::Buffered => {
                        debug!(
                            dst = %dst,
                            msg_type = %msg_type,
                            "[boot_buffer] buffered targeted P0 frame for late-attach"
                        );
                        return;
                    }
                    BootBufferPushOutcome::BufferedOverflowLogged => {
                        // 2026-05-17 — instrumented with src + from_name for
                        // BUG-PHASE-C-BOOT-BUFFER-GUARDIAN-OVERFLOW-PERSISTENT.
                        // The persistent overflow on dst=guardian survives the
                        // 545a6eca stale-alias fix; surface the publisher so
                        // we can trace which producer's HEARTBEATs are landing
                        // when guardian's subscriber should already match.
                        warn!(
                            dst = %dst,
                            msg_type = %msg_type,
                            src = %src,
                            from_name = %from_name,
                            "[boot_buffer] overflow on dst={} src={} from_name={} msg_type={} — oldest frame evicted (rate-limited 1/60s per dst). Subscriber attach delayed > expected boot window.",
                            dst, src, from_name, msg_type
                        );
                        return;
                    }
                    BootBufferPushOutcome::BufferedOverflowSilent => {
                        // Suppressed by rate-limiter — buffered, no log
                        return;
                    }
                    BootBufferPushOutcome::TypeNotBuffered => {
                        // Fall through to existing fanout behavior
                        // (will result in no delivery since target_keys
                        // is empty; existing logs / metrics unchanged).
                    }
                }
            }
        }

        // Drift bridge: emit under both names if applicable
        let names: Vec<String> = {
            let bridged = bridge_emit_names(&msg_type);
            if bridged.is_empty() {
                vec![msg_type.clone()]
            } else {
                bridged.into_iter().map(|s| s.to_string()).collect()
            }
        };

        // Resolve target subscribers by their registered `name` field
        // (NOT map key) OR matching alias. Map keys stay anon-N forever;
        // fanout filters by logical name(s). Publisher-skip applies ONLY
        // to broadcasts (dst="all") per SPEC §8.2 v1.4.0 D-SPEC-42 +
        // D-SPEC-52 (v1.7.3): "Targeted routing remains unaffected". A
        // worker MAY legitimately address itself by name when an in-process
        // consumer lives there (e.g. spirit_worker emitting META_CGN_SIGNAL
        // with dst="spirit" → MetaCGNConsumer in spirit). The aliases set
        // is populated by multi-name BUS_SUBSCRIBE (see
        // InboundEvent::Subscribe handler) and enables one connection to
        // be addressable as multiple dst names — used by Python titan_HCL
        // to register all its kernel-side proxy reply queues
        // (output_verifier_proxy, agency_proxy, …) on a single connection.
        let target_keys: Vec<String> = {
            let subs_guard = subs.lock().await;
            subs_guard
                .iter()
                .filter(|(map_key, _)| dst != "all" || map_key.as_str() != from_name)
                .filter(|(_, sub)| dst == "all" || sub.name == dst || sub.aliases.contains(dst))
                .map(|(map_key, _)| map_key.clone())
                .collect()
        };
        let target_names = target_keys; // alias for downstream loop body

        for target in target_names {
            let mut delivered_any = false;
            for emit_name in &names {
                let priority = get_spec(emit_name).priority;
                // For drift-bridge alias names, re-encode the bytes with the
                // bridged `type` field. For the canonical/original name we
                // re-use the publisher's bytes byte-identically (cheap path).
                let payload_bytes = if emit_name == &msg_type {
                    raw_bytes.clone()
                } else {
                    match rewrite_msg_type(&raw_bytes, emit_name) {
                        Ok(b) => b,
                        Err(e) => {
                            warn!(
                                err = ?e,
                                emit_name = %emit_name,
                                "drift-bridge rewrite failed; skipping bridged copy"
                            );
                            continue;
                        }
                    }
                };
                let envelope = Envelope {
                    msg_type: emit_name.clone(),
                    src: src.clone(),
                    priority,
                    payload: payload_bytes,
                };
                let mut subs_guard = subs.lock().await;
                if let Some(sub) = subs_guard.get_mut(&target) {
                    // D-SPEC-45 (SPEC v1.5.1 → v1.5.2 PATCH per §2.6 —
                    // closed-subscriber transient-state silent-skip):
                    // Subscribers marked `closed=true` by the heartbeat
                    // task (heartbeat.rs:108) are in tear-down transient
                    // state — the recv_task is still draining the dead
                    // TCP connection before the connection_handler at
                    // broker.rs:343 purges them from the map. They are
                    // NOT in the SPEC §8.2 v1.4.0 D-SPEC-42 row-3
                    // forbidden-regression state — they're a normal
                    // cleanup transient. Silent skip: no deliver, no
                    // warn, no drop counter. Without this guard, every
                    // broadcast fired during a stale-subscriber's purge
                    // window generates a WARN+drop log (observed
                    // 11K/min on T3 from a single dead anon subscriber).
                    if sub.closed {
                        continue;
                    }
                    // SPEC §8.2 v1.4.0 (D-SPEC-42) dispatch order for
                    // `dst="all"` broadcasts (closed subs already skipped above):
                    //   (1) reply_only=true → silent skip (no enqueue,
                    //       no warn, no drop counter; subscriber
                    //       declared it does not consume broadcasts).
                    //   (2) subscribed_topics non-empty AND msg_type
                    //       ∈ topics → deliver.
                    //   (3) subscribed_topics empty AND !reply_only →
                    //       WARN+drop (SPEC violation — caller must
                    //       declare topics OR reply_only=true).
                    //
                    // Targeted routing (dst != "all") was resolved by
                    // the outer `target_keys` filter and bypasses all
                    // three checks. Mirrors Python `BusSocketServer.publish`.
                    if dst == "all" {
                        if sub.reply_only {
                            // D-SPEC-42 (SPEC v1.4.0): reply-only
                            // subscribers do not receive broadcasts by
                            // SPEC. Silent skip — no log, no drop
                            // counter — this is the contracted path.
                            continue;
                        }
                        if !sub.has_subscribed {
                            // Pre-subscribe transient: the connection is up
                            // but has not yet sent any BUS_SUBSCRIBE (the
                            // normal connect→subscribe race). This is NOT the
                            // §8.2 forbidden regression — the connection
                            // simply hasn't declared intent yet. Silent skip
                            // (mirrors the D-SPEC-45 closed-subscriber skip):
                            // no enqueue, no warn, no drop counter. Without
                            // this, every broadcast fired during a freshly-
                            // connected worker's subscribe window logged a
                            // WARN+drop (observed at boot: SPHERE_PULSE /
                            // SPIRIT_STATE / MIND_STATE → anon-N).
                            continue;
                        }
                        if sub.subscribed_topics.is_empty() {
                            // SPEC §8.2 v1.4.0 forbidden regression:
                            // empty topics AND reply_only=false.
                            // Caller violates intent declaration; CI
                            // lockstep gate should have caught at
                            // commit time. Drop loudly so the field
                            // operator sees the regression.
                            warn!(
                                subscriber = %sub.name,
                                msg_type = %emit_name,
                                "[divine_bus] subscriber has empty broadcast_topics AND reply_only=false — broadcast dropped. Declare ModuleSpec.broadcast_topics OR set reply_only=true. Per SPEC §8.2 v1.4.0 + rFP_bus_reply_only_socket_broker_port §3."
                            );
                            continue;
                        }
                        if !sub.subscribed_topics.contains(emit_name) {
                            continue;
                        }
                    }
                    let _ = sub.publish(envelope);
                    delivered_any = true;
                }
            }
            if delivered_any {
                if let Some(notify) = notify_per_sub.lock().await.get(&target) {
                    notify.notify_one();
                }
            }
        }
    }
}

// ── Send loop helper that reads via the shared subs map ────────────────────

struct SubByName {
    subs: Arc<Mutex<SubscriberMap>>,
    name: String,
}

impl SubByName {
    fn new(subs: Arc<Mutex<SubscriberMap>>, name: String) -> Self {
        Self { subs, name }
    }
}

async fn run_send_loop_via_map(
    mut write_half: tokio::net::unix::OwnedWriteHalf,
    subs: Arc<Mutex<SubscriberMap>>,
    sub_name: String,
    notify: Arc<Notify>,
) {
    /// Mirrors Python `SEND_BATCH_THRESHOLD=5` from bus_socket.py:171.
    const SEND_BATCH_THRESHOLD: usize = 5;
    loop {
        notify.notified().await;
        loop {
            let popped: Vec<Envelope> = {
                let mut subs_guard = subs.lock().await;
                let sub = match subs_guard.get_mut(&sub_name) {
                    Some(s) => s,
                    None => return,
                };
                if sub.closed {
                    return;
                }
                sub.pop_for_send(SEND_BATCH_THRESHOLD)
            };
            if popped.is_empty() {
                break;
            }
            for env in popped {
                if let Err(e) = write_frame_to_half(&mut write_half, &env.payload).await {
                    debug!(name = %sub_name, err = ?e, "send loop: write failed; closing");
                    let mut subs_guard = subs.lock().await;
                    if let Some(sub) = subs_guard.get_mut(&sub_name) {
                        sub.closed = true;
                    }
                    return;
                }
            }
        }
    }
}

/// Write a length-prefixed frame to `OwnedWriteHalf`.
async fn write_frame_to_half(
    write_half: &mut tokio::net::unix::OwnedWriteHalf,
    payload: &[u8],
) -> std::io::Result<()> {
    use titan_core::frame::encode_frame;
    use tokio::io::AsyncWriteExt;
    let bytes = encode_frame(payload).map_err(|e| std::io::Error::other(format!("{e:?}")))?;
    write_half.write_all(&bytes).await?;
    Ok(())
}

// Reserve types/functions that are only used by tests/integration but kept
// in the public API for downstream crates.
#[allow(dead_code)]
const _UNUSED_PRIORITY: Option<Priority> = None;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn broker_constructible() {
        let _broker = BusBroker::new("T1", b"authkey".to_vec());
    }

    #[tokio::test]
    async fn broker_start_stop_clean() {
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("titan_bus_T1.sock");
        let mut broker = BusBroker::new("T1", b"authkey-32-bytes-shared-XXXXX!!".to_vec());
        broker.start(&sock_path).await.unwrap();
        assert_eq!(broker.subscriber_count().await, 0);
        broker.stop().await;
        // Socket file should be unlinked
        assert!(!sock_path.exists());
    }

    #[tokio::test]
    async fn broker_start_unlinks_stale_socket() {
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("titan_bus_T1.sock");
        // Pre-create a stale socket file
        std::fs::File::create(&sock_path).unwrap();
        assert!(sock_path.exists());

        let mut broker = BusBroker::new("T1", b"authkey-32-bytes-shared-XXXXX!!".to_vec());
        broker.start(&sock_path).await.unwrap();
        broker.stop().await;
    }
}
