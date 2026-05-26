//! server — Unix socket accept loop + HMAC challenge handshake +
//! per-connection async tasks.
//!
//! Byte-identical port of Python `BusSocketServer._accept_loop` +
//! `_handle_client` + `_handshake` + `_recv_loop` + `_send_loop`.
//!
//! # Handshake (SPEC §8 + B.2)
//!
//! 1. Server sends 32 random bytes (`FRAME_CHALLENGE_BYTES`) RAW (no length
//!    prefix; this is pre-frame).
//! 2. Client sends back `HMAC-SHA256(authkey, challenge)` (32 bytes raw).
//! 3. Server verifies with `constant_time_eq`. On mismatch → close.
//! 4. Both sides switch to length-prefix + msgpack framing.
//!
//! # Per-connection tasks
//!
//! After a successful handshake, the broker spawns:
//! - **recv task**: `read_frame()` loop; decodes msgpack header; routes to
//!   control handlers (BUS_SUBSCRIBE/UNSUBSCRIBE/PONG) or to the broker's
//!   fanout path.
//! - **send task**: waits on a `Notify`; drains the subscriber's ring;
//!   writes batched frames out the socket.
//!
//! Both tasks die when the connection closes; broker purges the subscriber
//! from the map.

use std::sync::Arc;

use rand::RngCore;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, watch, Mutex, Notify};
use tracing::{debug, warn};

use titan_core::constants::{FRAME_AUTH_TAG_BYTES, FRAME_CHALLENGE_BYTES};
use titan_core::frame::{
    compute_hmac, constant_time_eq, decode_length_prefix, encode_frame, FRAME_LENGTH_PREFIX_BYTES,
    FRAME_MAX_FRAME_BYTES,
};

use crate::message::{decode_bus_subscribe_payload, decode_header, MsgHeader};
use crate::subscriber::BrokerSubscriber;

/// Errors during connection handling.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// I/O failure during handshake or framing.
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    /// Handshake HMAC mismatch — closes the connection.
    #[error("handshake HMAC mismatch")]
    HandshakeMismatch,
    /// Frame too large (peer announced size > FRAME_MAX_FRAME_BYTES).
    #[error("frame size exceeds maximum: {actual}B > {max}B")]
    FrameTooLarge {
        /// Announced bytes.
        actual: u64,
        /// Configured maximum.
        max: u64,
    },
}

/// Server-side handshake (challenge + verify response).
///
/// Returns `Ok(())` on success; otherwise closes the connection (caller
/// drops the stream).
pub async fn perform_handshake(stream: &mut UnixStream, authkey: &[u8]) -> Result<(), ServerError> {
    // 1. Generate + send challenge (raw, no length prefix)
    let mut challenge = [0u8; FRAME_CHALLENGE_BYTES as usize];
    rand::rngs::OsRng.fill_bytes(&mut challenge);
    stream.write_all(&challenge).await?;

    // 2. Read client's response (raw, fixed length)
    let mut response = [0u8; FRAME_AUTH_TAG_BYTES as usize];
    stream.read_exact(&mut response).await?;

    // 3. Verify with constant_time_eq
    let expected = compute_hmac(authkey, &challenge);
    if !constant_time_eq(&response, &expected) {
        return Err(ServerError::HandshakeMismatch);
    }
    Ok(())
}

/// Read one length-prefixed frame from the stream. Returns the payload bytes.
pub async fn read_frame(stream: &mut UnixStream) -> Result<Vec<u8>, ServerError> {
    let mut prefix = [0u8; 4];
    stream.read_exact(&mut prefix).await?;
    let n = decode_length_prefix(&prefix).map_err(|_| ServerError::FrameTooLarge {
        actual: u32::from_le_bytes(prefix) as u64,
        max: FRAME_MAX_FRAME_BYTES,
    })? as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut payload = vec![0u8; n];
    stream.read_exact(&mut payload).await?;
    Ok(payload)
}

/// Write a length-prefixed frame to the stream.
pub async fn write_frame(stream: &mut UnixStream, payload: &[u8]) -> Result<(), ServerError> {
    let bytes = encode_frame(payload).map_err(|_| ServerError::FrameTooLarge {
        actual: payload.len() as u64,
        max: FRAME_MAX_FRAME_BYTES,
    })?;
    stream.write_all(&bytes).await?;
    Ok(())
}

// Suppress unused warning for the constant; it documents the protocol shape.
#[allow(dead_code)]
const _LENGTH_PREFIX_REMINDER: u64 = FRAME_LENGTH_PREFIX_BYTES;

// ─────────────────────────────────────────────────────────────────────────
// Per-connection recv + send tasks
// ─────────────────────────────────────────────────────────────────────────

/// Inbound control event detected by the recv task. The broker dispatches
/// these to per-subscriber state updates without going through the ring.
#[derive(Debug, Clone)]
pub enum InboundEvent {
    /// Client identified itself + subscribed to topics.
    Subscribe {
        /// Real subscriber name (replaces anonymous initial name).
        name: String,
        /// Topics to add to subscribed set.
        topics: Vec<String>,
        /// D-SPEC-42 (SPEC v1.4.0, 2026-05-12) subscriber-intent flag.
        /// When true, broker silently skips this subscriber from
        /// `dst="all"` broadcast fan-out (targeted routing still works).
        /// Connection-level — last value wins on multi-name subscribe.
        reply_only: bool,
    },
    /// Client unsubscribed from topics.
    Unsubscribe {
        /// Topics to remove from subscribed set.
        topics: Vec<String>,
    },
    /// Heartbeat reply.
    Pong,
    /// Generic message published by the client — broker fanouts to other
    /// subscribers.
    Publish {
        /// Decoded header (msg_type, src, dst).
        header: MsgHeader,
        /// Original raw msgpack bytes — broker forwards unchanged for
        /// byte-identical fanout.
        raw_bytes: Vec<u8>,
    },
}

/// Run the per-connection recv loop until EOF or error.
///
/// Sends events to the broker via `inbound_tx`. Broker holds the receiver
/// + dispatches to per-subscriber state.
pub async fn run_recv_loop(
    mut read_half: tokio::net::unix::OwnedReadHalf,
    sub_name: String,
    inbound_tx: mpsc::UnboundedSender<(String, InboundEvent)>,
    // SPEC §8.0.quat / D-SPEC-131 (2026-05-26): broker-side close-state
    // channel (level-triggered). Set ONCE by
    // `BrokerSubscriber::signal_close()` when the subscriber is marked
    // closed (heartbeat timeout / send-loop write failure / explicit
    // shutdown). `wait_for(|v| *v)` resolves immediately if already
    // true at subscribe time, so the H1 race (signal_close fires before
    // recv_loop reaches its select) cannot lose the signal.
    //
    // **Distinct from the data-wake `notify`** — recv_loop MUST NOT
    // listen to the data-wake notify (SPEC §8.0.quat invariant 2). The
    // D-SPEC-130 partial fix re-used the data-wake notify for shutdown
    // signaling; that caused recv_loop to be probabilistically woken
    // by every fanout's `notify_one()` (undefined choice-of-waiter
    // semantics) → recv_loop exited thinking it was a shutdown → fleet
    // cascade. D-SPEC-131 separates the two primitives.
    mut close_rx: watch::Receiver<bool>,
) {
    // We re-create a UnixStream-like read API on the half. tokio's
    // OwnedReadHalf supports AsyncRead but not the full UnixStream API
    // expected by `read_frame`. Cleaner fix: have `read_frame` take any
    // AsyncRead — generic. For C2-2.b we use a small helper.
    use tokio::io::AsyncReadExt;
    loop {
        // Read length prefix (4 bytes). Wrapped in tokio::select! with
        // the close-state receiver so a broker-side close request (e.g.
        // heartbeat timeout) wakes us out of the blocking read.
        let mut prefix = [0u8; 4];
        let prefix_result = tokio::select! {
            r = read_half.read_exact(&mut prefix) => r,
            // Wrap `wait_for` in an inline async block so its `Ref<'_, bool>`
            // (carries an `RwLockReadGuard`, not `Send`) is dropped INSIDE
            // the block and the select's `Out` enum only stores `()`.
            _ = async { let _ = close_rx.wait_for(|v| *v).await; } => {
                // SPEC §8.0.quat exit-reason: authoritative close-state
                // transition observed by recv_loop. Surfaced at INFO per
                // SPEC error-cascade discipline (Maker 2026-05-25,
                // extended 2026-05-26) so kernel logs retain a single
                // unambiguous trace of every connection exit reason.
                tracing::info!(
                    name = %sub_name,
                    reason = "broker_close_signal",
                    "recv loop: exiting on close-state signal from broker"
                );
                return;
            }
        };
        if prefix_result.is_err() {
            // Promoted DEBUG→INFO: peer-close is normal but the kernel
            // log MUST retain a marker (SPEC error-cascade discipline).
            tracing::info!(
                name = %sub_name,
                reason = "peer_eof_prefix",
                "recv loop: peer closed (EOF before length-prefix)"
            );
            break;
        }
        let n = match decode_length_prefix(&prefix) {
            Ok(n) => n as usize,
            Err(e) => {
                // Promoted WARN→ERROR with structured fields: a
                // malformed length-prefix means the wire protocol is
                // broken (auth bypass attempt? client bug? memory
                // corruption?) — operator MUST see this.
                tracing::error!(
                    name = %sub_name,
                    err = ?e,
                    reason = "malformed_length_prefix",
                    "recv loop: ERROR — invalid length prefix; closing connection"
                );
                break;
            }
        };
        let mut payload = vec![0u8; n];
        if n > 0 {
            let payload_result = tokio::select! {
                r = read_half.read_exact(&mut payload) => r,
                _ = async { let _ = close_rx.wait_for(|v| *v).await; } => {
                    tracing::info!(
                        name = %sub_name,
                        reason = "broker_close_signal_mid_frame",
                        "recv loop: exiting on close-state signal mid-frame"
                    );
                    return;
                }
            };
            if payload_result.is_err() {
                tracing::info!(
                    name = %sub_name,
                    reason = "peer_eof_mid_frame",
                    "recv loop: peer closed mid-frame"
                );
                break;
            }
        }

        // Decode header + classify
        let header = match decode_header(&payload) {
            Ok(h) => h,
            Err(e) => {
                warn!(name = %sub_name, err = ?e, len = payload.len(), "recv loop: malformed frame; closing");
                break;
            }
        };

        let event = match header.msg_type.as_deref() {
            Some("BUS_SUBSCRIBE") => {
                // SPEC §8.2 (v1.3.0): BUS_SUBSCRIBE payload is
                // `{name: str, topics: [str]}`. The CANONICAL subscriber
                // name is `payload.name` — NOT `header.src`. Multi-name
                // BUS_SUBSCRIBE relies on this: a single connection can
                // register under multiple names by sending repeated
                // BUS_SUBSCRIBE frames over the SAME connection, each
                // with a different `payload.name`. Pre-v1.3.0 code that
                // wrote `header.src` as the name worked for single-name
                // callers by coincidence (Python BusSocketClient sets
                // both header.src and payload.name to self.name) but
                // dropped multi-name aliases on the floor (the broker
                // saw N frames all naming "titan_HCL"). Falls back to
                // header.src if payload.name is missing to preserve
                // backward-compat with any client that omits it.
                // D-SPEC-42 (SPEC v1.4.0): decode_bus_subscribe_payload
                // returns a 3-tuple including reply_only intent. Backward
                // compatible: pre-v1.4.0 clients omit the field, decoder
                // returns reply_only=false (broadcast-consumer intent).
                let (payload_name, payload_topics, payload_reply_only) =
                    match decode_bus_subscribe_payload(&payload) {
                        Ok(t) => t,
                        Err(e) => {
                            warn!(name = %sub_name, err = ?e, "recv loop: BUS_SUBSCRIBE payload decode failed; falling back to header.src");
                            (None, Vec::new(), false)
                        }
                    };
                let resolved_name = payload_name
                    .filter(|s| !s.is_empty())
                    .or_else(|| header.src.clone())
                    .unwrap_or_default();
                InboundEvent::Subscribe {
                    name: resolved_name,
                    topics: payload_topics,
                    reply_only: payload_reply_only,
                }
            }
            Some("BUS_UNSUBSCRIBE") => InboundEvent::Unsubscribe { topics: Vec::new() },
            Some("BUS_PONG") => InboundEvent::Pong,
            Some(_) => InboundEvent::Publish {
                header,
                raw_bytes: payload,
            },
            None => {
                warn!(name = %sub_name, "recv loop: missing type; closing");
                break;
            }
        };

        if inbound_tx.send((sub_name.clone(), event)).is_err() {
            debug!(name = %sub_name, "recv loop: broker dropped inbound channel");
            break;
        }
    }
}

// Send loop lives in `broker::run_send_loop_via_map` since it reads via the
// shared subscriber map (matches Python `BusSocketServer._send_loop` design).

// Suppress unused-import warning when this module is built without a broker
// driving it (tests etc.). `Notify` retained for backward-compat with any
// downstream crate still importing it from titan_bus::server; the actual
// shutdown primitive used by run_recv_loop is `watch::Receiver<bool>`
// (SPEC §8.0.quat / D-SPEC-131).
#[allow(dead_code)]
const _UNUSED: Option<(Arc<Mutex<BrokerSubscriber>>, Arc<Notify>)> = None;

// ─────────────────────────────────────────────────────────────────────────
// Tests (handshake-level only; integration test in tests/integration.rs)
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn handshake_succeeds_with_correct_authkey() {
        let (server, client) = tokio::net::UnixStream::pair().unwrap();
        let authkey = b"shared-secret-32-bytes-exactly!!".to_vec();

        // Spawn server side
        let authkey_srv = authkey.clone();
        let server_task = tokio::spawn(async move {
            let mut server = server;
            perform_handshake(&mut server, &authkey_srv).await
        });

        // Client side: read challenge, send HMAC response
        let authkey_cli = authkey.clone();
        let client_task = tokio::spawn(async move {
            let mut client = client;
            let mut challenge = [0u8; FRAME_CHALLENGE_BYTES as usize];
            client.read_exact(&mut challenge).await.unwrap();
            let response = compute_hmac(&authkey_cli, &challenge);
            client.write_all(&response).await.unwrap();
            // Hold the stream open while server side completes
            let mut buf = [0u8; 1];
            let _ = tokio::time::timeout(Duration::from_millis(100), client.read(&mut buf)).await;
        });

        let result = server_task.await.unwrap();
        let _ = client_task.await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn handshake_fails_with_wrong_authkey() {
        let (server, client) = tokio::net::UnixStream::pair().unwrap();
        let server_key = b"server-key-32-bytes-exactlyXXXXX".to_vec();
        let client_key = b"WRONG-KEY-32-bytes-exactly!!ABCD".to_vec();

        let server_task = tokio::spawn(async move {
            let mut server = server;
            perform_handshake(&mut server, &server_key).await
        });

        let client_task = tokio::spawn(async move {
            let mut client = client;
            let mut challenge = [0u8; FRAME_CHALLENGE_BYTES as usize];
            client.read_exact(&mut challenge).await.unwrap();
            let response = compute_hmac(&client_key, &challenge);
            client.write_all(&response).await.unwrap();
            let mut buf = [0u8; 1];
            let _ = tokio::time::timeout(Duration::from_millis(100), client.read(&mut buf)).await;
        });

        let result = server_task.await.unwrap();
        let _ = client_task.await;
        assert!(matches!(result, Err(ServerError::HandshakeMismatch)));
    }

    #[tokio::test]
    async fn frame_round_trip_via_pair() {
        let (mut server, mut client) = tokio::net::UnixStream::pair().unwrap();
        let payload = b"hello, world";

        // Client writes
        let payload_clone = payload.to_vec();
        let writer = tokio::spawn(async move {
            write_frame(&mut client, &payload_clone).await.unwrap();
        });

        // Server reads
        let received = read_frame(&mut server).await.unwrap();
        assert_eq!(received, payload);
        writer.await.unwrap();
    }

    #[test]
    fn inbound_event_variants_are_constructible() {
        // Smoke: ensure the enum variants can be built with realistic data
        let _ = InboundEvent::Subscribe {
            name: "test".into(),
            topics: vec!["BODY_STATE".into()],
            reply_only: false,
        };
        let _ = InboundEvent::Subscribe {
            name: "rpc_reply_queue".into(),
            topics: vec![],
            reply_only: true,
        };
        let _ = InboundEvent::Unsubscribe {
            topics: vec!["BODY_STATE".into()],
        };
        let _ = InboundEvent::Pong;
        let _ = InboundEvent::Publish {
            header: MsgHeader::default(),
            raw_bytes: vec![0; 16],
        };
    }
}
