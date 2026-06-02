//! api_reload_subscriber — the kernel's FIRST inbound bus subscriber.
//!
//! SPEC §11.B.5 / D-SPEC-149 — gives the kernel a deliberate, zero-downtime
//! L3 api code-reload path. This task subscribes the `KERNEL_API_RELOAD_REQUEST`
//! bus command (§8.1) over a `BusClient` connected to the kernel's OWN broker
//! socket, and forwards each command onto the api `watch_loop`'s reload channel
//! (`KernelChildSupervisor::api_reload_sender`).
//!
//! Per Preamble §G18 the trigger is a *command* → correctly travels the bus
//! (the health-gate, by contrast, is *state* → read from SHM in P3). This is
//! the kernel's first inbound bus subscriber — every other kernel bus touch is
//! outbound (`broker_publisher` / `fastbus_publisher`). It is the seed of the
//! broader D9.2 Rust-Guardian control plane.
//!
//! Resilience: connect + subscribe with retry; on disconnect, reconnect. The
//! task only exits on kernel shutdown (`shutdown.notified()`).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, Notify};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use titan_bus::{BusClient, InboundEvent};

use crate::kernel_supervisor::ApiReloadCommand;

/// The bus topic the kernel subscribes (SPEC §8.1, D-SPEC-149).
const TOPIC: &str = "KERNEL_API_RELOAD_REQUEST";

/// Our subscriber identity on the bus. Distinct from the kernel-internal
/// publisher labels (`kernel:clocks`, `kernel:supervisor`) so the broker's
/// echo-skip + audit logs disambiguate the inbound control channel.
const SUBSCRIBER_NAME: &str = "kernel";

/// Reconnect / retry backoff (the kernel's own broker is local + always up in
/// steady state, so a short fixed delay is ample — this only paces a boot-order
/// race or a transient socket churn).
const RECONNECT_DELAY: Duration = Duration::from_secs(1);

/// Spawn the kernel's inbound api-reload bus subscriber. Returns the task
/// handle (awaited / aborted at shutdown).
pub fn spawn_api_reload_subscriber(
    bus_socket: PathBuf,
    authkey: Vec<u8>,
    reload_tx: mpsc::Sender<ApiReloadCommand>,
    shutdown: Arc<Notify>,
) -> JoinHandle<()> {
    tokio::spawn(async move { run(bus_socket, authkey, reload_tx, shutdown).await })
}

async fn run(
    bus_socket: PathBuf,
    authkey: Vec<u8>,
    reload_tx: mpsc::Sender<ApiReloadCommand>,
    shutdown: Arc<Notify>,
) {
    loop {
        // ── Connect ───────────────────────────────────────────────────────
        let client = match BusClient::connect(&bus_socket, &authkey, SUBSCRIBER_NAME).await {
            Ok(c) => c,
            Err(e) => {
                warn!(
                    err = ?e,
                    path = ?bus_socket,
                    "api reload subscriber: connect to own broker failed; retrying"
                );
                if sleep_or_shutdown(&shutdown).await {
                    return;
                }
                continue;
            }
        };

        // ── Subscribe ─────────────────────────────────────────────────────
        if let Err(e) = client.subscribe(&[TOPIC]).await {
            warn!(err = ?e, "api reload subscriber: subscribe failed; reconnecting");
            client.shutdown().await;
            if sleep_or_shutdown(&shutdown).await {
                return;
            }
            continue;
        }
        info!(
            event = "KERNEL_API_RELOAD_SUBSCRIBER_READY",
            topic = TOPIC,
            "kernel inbound bus subscriber attached (first inbound subscriber — §11.B.5)"
        );

        // ── Receive loop ──────────────────────────────────────────────────
        loop {
            tokio::select! {
                _ = shutdown.notified() => {
                    info!("api reload subscriber: shutdown — closing");
                    client.shutdown().await;
                    return;
                }
                ev = client.recv() => {
                    match ev {
                        Some(InboundEvent::Message { msg_type, raw_bytes, .. })
                            if msg_type == TOPIC =>
                        {
                            let (reason, requested_by) = decode_reload_payload(&raw_bytes);
                            info!(
                                event = "KERNEL_API_RELOAD_REQUEST_RECV",
                                reason = %reason,
                                requested_by = %requested_by,
                                "received api reload request; forwarding to api watch loop"
                            );
                            if let Err(e) = reload_tx
                                .send(ApiReloadCommand { reason, requested_by })
                                .await
                            {
                                // Receiver gone = api watch task exited (kernel
                                // tearing down). Nothing actionable; log + keep
                                // serving the bus until shutdown fires.
                                error!(
                                    err = ?e,
                                    "api reload subscriber: api watch loop receiver closed; \
                                     reload not delivered"
                                );
                            }
                        }
                        // Other topics shouldn't arrive (we only subscribed one);
                        // ignore defensively.
                        Some(InboundEvent::Message { .. }) => {}
                        Some(InboundEvent::Disconnected { reason }) => {
                            warn!(reason = %reason, "api reload subscriber disconnected; reconnecting");
                            break;
                        }
                        None => {
                            warn!("api reload subscriber: event channel closed; reconnecting");
                            break;
                        }
                    }
                }
            }
        }

        // Reconnect after a brief pause (unless shutting down).
        client.shutdown().await;
        if sleep_or_shutdown(&shutdown).await {
            return;
        }
    }
}

/// Sleep `RECONNECT_DELAY`, or return `true` immediately if shutdown fired.
async fn sleep_or_shutdown(shutdown: &Arc<Notify>) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(RECONNECT_DELAY) => false,
        _ = shutdown.notified() => true,
    }
}

/// Extract `{reason, requested_by}` from a `KERNEL_API_RELOAD_REQUEST` envelope
/// (SPEC §8.1 payload `{ts, reason, requested_by}`). Missing fields fall back to
/// sane defaults so a malformed command still produces an auditable reload.
fn decode_reload_payload(envelope_bytes: &[u8]) -> (String, String) {
    let mut reason = String::from("unspecified");
    let mut requested_by = String::from("unknown");
    if let Some(rmpv::Value::Map(entries)) = titan_bus::client::extract_payload(envelope_bytes) {
        for (k, v) in entries {
            if let rmpv::Value::String(ks) = &k {
                match ks.as_str() {
                    Some("reason") => {
                        if let Some(s) = v.as_str() {
                            reason = s.to_string();
                        }
                    }
                    Some("requested_by") => {
                        if let Some(s) = v.as_str() {
                            requested_by = s.to_string();
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    (reason, requested_by)
}

#[cfg(test)]
mod tests {
    use super::*;
    use titan_bus::message::encode_simple;

    #[test]
    fn decode_reload_payload_extracts_fields() {
        let payload = rmpv::Value::Map(vec![
            (
                rmpv::Value::String("ts".into()),
                rmpv::Value::F64(1779000000.0),
            ),
            (
                rmpv::Value::String("reason".into()),
                rmpv::Value::String("deploy dashboard.py allowlist edit".into()),
            ),
            (
                rmpv::Value::String("requested_by".into()),
                rmpv::Value::String("operator@t1".into()),
            ),
        ]);
        let envelope =
            encode_simple(TOPIC, Some("operator"), Some("kernel"), Some(payload)).unwrap();
        let (reason, requested_by) = decode_reload_payload(&envelope);
        assert_eq!(reason, "deploy dashboard.py allowlist edit");
        assert_eq!(requested_by, "operator@t1");
    }

    #[test]
    fn decode_reload_payload_defaults_on_missing_fields() {
        let payload = rmpv::Value::Map(vec![(
            rmpv::Value::String("ts".into()),
            rmpv::Value::F64(1779000000.0),
        )]);
        let envelope =
            encode_simple(TOPIC, Some("operator"), Some("kernel"), Some(payload)).unwrap();
        let (reason, requested_by) = decode_reload_payload(&envelope);
        assert_eq!(reason, "unspecified");
        assert_eq!(requested_by, "unknown");
    }

    #[test]
    fn decode_reload_payload_handles_garbage() {
        let (reason, requested_by) = decode_reload_payload(b"not-msgpack");
        assert_eq!(reason, "unspecified");
        assert_eq!(requested_by, "unknown");
    }

    /// Full P2 path: a `KERNEL_API_RELOAD_REQUEST` published to a real broker
    /// is received by the kernel subscriber and forwarded as an
    /// `ApiReloadCommand` onto the api watch_loop's reload channel. This proves
    /// "the command reaches the loop" (rFP §6 P2 acceptance).
    #[tokio::test]
    async fn reload_request_published_reaches_reload_channel() {
        use std::time::Duration;
        use titan_bus::{BusBroker, BusClient};
        use tokio::sync::{mpsc, Notify};

        const AUTHKEY: &[u8] = b"shared-secret-32-bytes-exactly!!";

        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("titan_bus_TEST.sock");

        // Stand up a real broker (the kernel's own broker, in production).
        let mut broker = BusBroker::new("TEST", AUTHKEY.to_vec());
        broker.start(&sock).await.unwrap();

        // Spawn the kernel subscriber against it.
        let (tx, mut rx) = mpsc::channel::<ApiReloadCommand>(8);
        let shutdown = Arc::new(Notify::new());
        let handle =
            spawn_api_reload_subscriber(sock.clone(), AUTHKEY.to_vec(), tx, shutdown.clone());

        // Allow connect + BUS_SUBSCRIBE to land.
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(
            broker.subscriber_count().await >= 1,
            "kernel subscriber should be connected"
        );

        // Publish a reload request as the operator surface would.
        let publisher = BusClient::connect(&sock, AUTHKEY, "operator")
            .await
            .unwrap();
        let payload = rmpv::Value::Map(vec![
            (
                rmpv::Value::String("ts".into()),
                rmpv::Value::F64(1779000000.0),
            ),
            (
                rmpv::Value::String("reason".into()),
                rmpv::Value::String("deploy dashboard.py edit".into()),
            ),
            (
                rmpv::Value::String("requested_by".into()),
                rmpv::Value::String("operator@t1".into()),
            ),
        ]);
        publisher
            .publish("KERNEL_API_RELOAD_REQUEST", Some("all"), Some(payload))
            .await
            .unwrap();

        // The subscriber must forward an ApiReloadCommand onto the channel.
        let cmd = tokio::time::timeout(Duration::from_secs(3), rx.recv())
            .await
            .expect("reload command should arrive within 3s")
            .expect("reload channel should be open");
        assert_eq!(cmd.reason, "deploy dashboard.py edit");
        assert_eq!(cmd.requested_by, "operator@t1");

        // Clean shutdown.
        publisher.shutdown().await;
        shutdown.notify_waiters();
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
        broker.stop().await;
    }
}
