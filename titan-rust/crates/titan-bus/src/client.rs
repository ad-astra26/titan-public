//! bus_client — Connect to the kernel-owned main bus broker as a client.
//!
//! Per SPEC §6.3 (HKDF authkey + HMAC handshake) + §8.0 priority lanes +
//! §8.6 SPHERE_PULSE row + §9.A unified-spirit-rs row.
//!
//! Wire protocol (mirroring `titan-bus::server::perform_handshake` server
//! side):
//! 1. Connect to `/tmp/titan_bus_<id>.sock` via `tokio::net::UnixStream`.
//! 2. Server sends a 32-byte challenge (raw, no length prefix).
//! 3. Client computes `HMAC-SHA256(authkey, challenge)` (32 bytes raw)
//!    and writes back to the socket.
//! 4. Server verifies in constant time. On success, framed messages flow.
//! 5. Client sends `BUS_SUBSCRIBE` with `src=<our_name>` + msgpack payload
//!    `{"topics": [...]}`. Broker holds the subscription state.
//! 6. Client receives length-prefixed msgpack frames; the recv loop
//!    auto-responds to `BUS_PING` with `BUS_PONG` directly via the
//!    shared write half — `BUS_PING` is never forwarded to the caller's
//!    events stream. Per SPEC §3.1 D10 + §10.B; closes rFP chunk 9D.
//!
//! Designed to be reused by other Rust binaries (substrate, daemons) —
//! kept in this crate for C-S4 since the trinity daemons get their own
//! bus client wiring in C-S5/C-S6 (master plan §10.5–§10.6).
//!
//! Usage:
//! ```ignore
//! let client = BusClient::connect(&path, &authkey, "unified-spirit").await?;
//! client.subscribe(&["SPHERE_PULSE", "KERNEL_SHUTDOWN_ANNOUNCE"]).await?;
//! let mut events = client.events();
//! while let Some(event) = events.recv().await {
//!     match event { InboundEvent::Message { msg_type, raw_bytes, .. } => /* dispatch */ }
//! }
//! ```

use std::path::Path;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::message::{decode_header, encode_simple, MsgHeader};
use titan_core::constants::{FRAME_AUTH_TAG_BYTES, FRAME_CHALLENGE_BYTES};
use titan_core::frame::{compute_hmac, decode_length_prefix, encode_frame, FRAME_MAX_FRAME_BYTES};

/// Shared write half so the recv loop can auto-PONG on `BUS_PING`
/// without bouncing through the caller. Per SPEC §3.1 D10 + §10.B —
/// subscriber MUST pong; broker drops connection at 15s of missed pongs.
type SharedWrite = Arc<Mutex<tokio::net::unix::OwnedWriteHalf>>;

/// Errors during bus client connect / handshake / I/O.
#[derive(Debug, thiserror::Error)]
pub enum BusClientError {
    /// Could not connect to the broker socket. Likely the kernel hasn't
    /// bound yet (boot-order race) or path is wrong.
    #[error("connect to {path}: {source}")]
    Connect {
        /// Socket path attempted.
        path: String,
        /// Underlying io error.
        #[source]
        source: std::io::Error,
    },
    /// Handshake response too short or i/o error during handshake.
    #[error("handshake io: {0}")]
    HandshakeIo(#[from] std::io::Error),
    /// Frame encoding (publish) errored — payload exceeds size cap.
    #[error("frame encode: {0}")]
    FrameEncode(#[from] titan_core::frame::FrameError),
    /// msgpack encoding errored (rare — only for malformed inputs).
    #[error("msgpack encode: {0}")]
    MsgEncode(#[from] crate::message::MsgError),
    /// Subscriber channel dropped — the recv task exited.
    #[error("event channel closed (recv task exited)")]
    EventChannelClosed,
}

/// One inbound event surfaced to the caller.
#[derive(Debug, Clone)]
pub enum InboundEvent {
    /// A bus message arrived. Caller filters on `msg_type`.
    Message {
        /// Message type (e.g. `"SPHERE_PULSE"`).
        msg_type: String,
        /// Source identifier (publisher name).
        src: Option<String>,
        /// Destination — usually `Some("all")` for fanout.
        dst: Option<String>,
        /// Original msgpack bytes — caller decodes per message type.
        raw_bytes: Vec<u8>,
    },
    /// Broker closed the connection / network error. Caller may
    /// reconnect via `BusClient::connect`.
    Disconnected {
        /// Reason text for telemetry.
        reason: String,
    },
}

/// Connected bus client. Splits the socket into a shared write half
/// (held by `BusClient` for publish/subscribe + by the recv task for
/// auto-PONG) and read half (driven by an internal task that emits
/// `InboundEvent`s on `events_rx`).
pub struct BusClient {
    write_half: SharedWrite,
    events_rx: tokio::sync::Mutex<mpsc::UnboundedReceiver<InboundEvent>>,
    recv_task: tokio::sync::Mutex<Option<JoinHandle<()>>>,
    name: String,
}

impl BusClient {
    /// Connect, perform HMAC handshake, spawn the recv loop. Caller is
    /// responsible for calling [`BusClient::subscribe`] before any
    /// publishes since subscription identifies us to the broker.
    pub async fn connect(
        socket_path: &Path,
        authkey: &[u8],
        client_name: &str,
    ) -> Result<Self, BusClientError> {
        let mut stream =
            UnixStream::connect(socket_path)
                .await
                .map_err(|source| BusClientError::Connect {
                    path: socket_path.display().to_string(),
                    source,
                })?;

        // Handshake (mirror `titan_bus::server::perform_handshake` client side).
        let mut challenge = [0u8; FRAME_CHALLENGE_BYTES as usize];
        stream.read_exact(&mut challenge).await?;
        let response = compute_hmac(authkey, &challenge);
        debug_assert_eq!(response.len(), FRAME_AUTH_TAG_BYTES as usize);
        stream.write_all(&response).await?;

        // Split the stream so writes + reads can happen concurrently.
        let (read_half, write_half) = stream.into_split();
        let write_shared: SharedWrite = Arc::new(Mutex::new(write_half));

        // Spawn the recv loop. Events flow into events_rx; the recv loop
        // also auto-PONGs on BUS_PING via a clone of the shared write half.
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let write_for_recv = Arc::clone(&write_shared);
        let name_for_recv = client_name.to_string();
        let recv_task = tokio::spawn(run_recv_loop(
            read_half,
            events_tx,
            write_for_recv,
            name_for_recv,
        ));

        Ok(BusClient {
            write_half: write_shared,
            events_rx: tokio::sync::Mutex::new(events_rx),
            recv_task: tokio::sync::Mutex::new(Some(recv_task)),
            name: client_name.to_string(),
        })
    }

    /// Send `BUS_SUBSCRIBE` to the broker with our `src` name + topic list.
    /// Topics are encoded as a msgpack array under key `"topics"`.
    pub async fn subscribe(&self, topics: &[&str]) -> Result<(), BusClientError> {
        let payload = encode_subscribe_payload(topics)?;
        let envelope = encode_simple(
            "BUS_SUBSCRIBE",
            Some(self.name.as_str()),
            Some("broker"),
            Some(&payload),
        )?;
        let frame = encode_frame(&envelope)?;
        let mut w = self.write_half.lock().await;
        w.write_all(&frame).await?;
        Ok(())
    }

    /// Publish a message. Caller passes the full msgpack envelope (already
    /// includes `type`, `src`, `dst`, `payload` keys per `encode_simple`).
    pub async fn publish_envelope(&self, envelope_bytes: &[u8]) -> Result<(), BusClientError> {
        let frame = encode_frame(envelope_bytes)?;
        let mut w = self.write_half.lock().await;
        w.write_all(&frame).await?;
        Ok(())
    }

    /// Convenience: publish a message with type + dst + raw msgpack
    /// payload. `src` defaults to our client name.
    pub async fn publish(
        &self,
        msg_type: &str,
        dst: Option<&str>,
        payload: Option<&[u8]>,
    ) -> Result<(), BusClientError> {
        let envelope = encode_simple(msg_type, Some(self.name.as_str()), dst, payload)?;
        self.publish_envelope(&envelope).await
    }

    /// Internal: respond `BUS_PONG` to broker `BUS_PING`. Called by the
    /// recv loop via the events channel — caller's poll loop pings us.
    pub async fn send_pong(&self) -> Result<(), BusClientError> {
        let envelope = encode_simple("BUS_PONG", Some(self.name.as_str()), Some("broker"), None)?;
        self.publish_envelope(&envelope).await
    }

    /// Receive the next inbound event. Returns `None` when the recv task
    /// exits (broker closed, network error). Caller filters on msg_type.
    pub async fn recv(&self) -> Option<InboundEvent> {
        let mut rx = self.events_rx.lock().await;
        rx.recv().await
    }

    /// Shut down the recv task explicitly. Idempotent.
    pub async fn shutdown(&self) {
        let mut task_lock = self.recv_task.lock().await;
        if let Some(task) = task_lock.take() {
            task.abort();
            let _ = task.await;
        }
    }

    /// Client name (matches `src` on outbound envelopes).
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Encode `{"topics": [topic1, topic2, ...]}` as msgpack bytes.
fn encode_subscribe_payload(topics: &[&str]) -> Result<Vec<u8>, BusClientError> {
    let mut entries = Vec::with_capacity(1);
    let topic_values: Vec<rmpv::Value> = topics
        .iter()
        .map(|t| rmpv::Value::String((*t).into()))
        .collect();
    entries.push((
        rmpv::Value::String("topics".into()),
        rmpv::Value::Array(topic_values),
    ));
    let map = rmpv::Value::Map(entries);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &map).map_err(|e| {
        BusClientError::MsgEncode(crate::message::MsgError::Decode(format!("{e:?}")))
    })?;
    Ok(out)
}

/// Recv loop — reads length-prefixed frames, decodes headers, dispatches
/// to events_tx. Auto-handles `BUS_PING` by writing `BUS_PONG` directly
/// via the shared write half (no caller intervention required); BUS_PING
/// is consumed and never forwarded to events_tx. Per SPEC §3.1 D10 +
/// §10.B 3-layer heartbeat hierarchy. Closes rFP_phase_c_close_all_runtime_gaps
/// chunk 9D — without auto-PONG every Rust subscriber missed pongs and the
/// broker dropped the connection at 15s, cascade-killing the kernel within
/// 25-30s of every boot.
async fn run_recv_loop(
    mut read_half: tokio::net::unix::OwnedReadHalf,
    events_tx: mpsc::UnboundedSender<InboundEvent>,
    write_half: SharedWrite,
    name: String,
) {
    loop {
        // Read 4-byte length prefix
        let mut prefix = [0u8; 4];
        if let Err(e) = read_half.read_exact(&mut prefix).await {
            let _ = events_tx.send(InboundEvent::Disconnected {
                reason: format!("read length prefix: {e}"),
            });
            return;
        }
        let n = match decode_length_prefix(&prefix) {
            Ok(v) => v,
            Err(e) => {
                let _ = events_tx.send(InboundEvent::Disconnected {
                    reason: format!("frame too large: {e}"),
                });
                return;
            }
        };
        if n as u64 > FRAME_MAX_FRAME_BYTES {
            let _ = events_tx.send(InboundEvent::Disconnected {
                reason: format!("frame too large: {n} bytes"),
            });
            return;
        }
        if n == 0 {
            // Empty frame — broker shouldn't send these but be lenient.
            continue;
        }

        let mut payload = vec![0_u8; n as usize];
        if let Err(e) = read_half.read_exact(&mut payload).await {
            let _ = events_tx.send(InboundEvent::Disconnected {
                reason: format!("read payload: {e}"),
            });
            return;
        }

        // Decode header
        let header: MsgHeader = match decode_header(&payload) {
            Ok(h) => h,
            Err(e) => {
                warn!(err = ?e, "bus_client: malformed envelope; dropping");
                continue;
            }
        };

        let msg_type = match header.msg_type {
            Some(t) => t,
            None => {
                debug!("bus_client: envelope missing 'type' field; dropping");
                continue;
            }
        };

        // Auto-PONG: short-circuit BUS_PING by writing a BUS_PONG envelope
        // directly via the shared write half. Per SPEC §3.1 D10 + §10.B —
        // subscriber MUST pong; broker drops connection at 15s of missed
        // pongs (3-missed-pings rule). BUS_PING is NOT forwarded to caller.
        if msg_type == "BUS_PING" {
            match encode_simple("BUS_PONG", Some(name.as_str()), Some("broker"), None) {
                Ok(envelope) => match encode_frame(&envelope) {
                    Ok(frame) => {
                        let mut w = write_half.lock().await;
                        if let Err(e) = w.write_all(&frame).await {
                            // PONG write failed — broker likely closed.
                            // Surface as Disconnected so the caller can
                            // reconnect; do not fight a dead socket.
                            let _ = events_tx.send(InboundEvent::Disconnected {
                                reason: format!("BUS_PONG write failed: {e}"),
                            });
                            return;
                        }
                    }
                    Err(e) => {
                        warn!(err = ?e, "bus_client: failed to frame BUS_PONG");
                    }
                },
                Err(e) => {
                    warn!(err = ?e, "bus_client: failed to encode BUS_PONG");
                }
            }
            continue;
        }

        if events_tx
            .send(InboundEvent::Message {
                msg_type,
                src: header.src,
                dst: header.dst,
                raw_bytes: payload,
            })
            .is_err()
        {
            // Receiver dropped — caller exited; clean shutdown.
            return;
        }
    }
}

/// Helper for tests + integration: extract a typed payload from a
/// msgpack envelope's `"payload"` field (binary blob nested inside the
/// outer map). Returns the raw payload bytes that callers can decode
/// further per message-type schema.
pub fn extract_payload(envelope_bytes: &[u8]) -> Option<Vec<u8>> {
    let value: rmpv::Value =
        rmpv::decode::read_value(&mut std::io::Cursor::new(envelope_bytes)).ok()?;
    if let rmpv::Value::Map(entries) = value {
        for (k, v) in entries {
            if let rmpv::Value::String(s) = &k {
                if s.as_str() == Some("payload") {
                    if let rmpv::Value::Binary(b) = v {
                        return Some(b);
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;
    use titan_core::frame::compute_hmac;

    /// Helper: run a fake broker server that completes one handshake
    /// against the supplied authkey, then immediately closes the socket.
    /// Returns the socket path.
    async fn fake_broker_handshake_only(
        authkey: Vec<u8>,
    ) -> (tempfile::TempDir, std::path::PathBuf, JoinHandle<()>) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bus.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let task = tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut challenge = [0u8; FRAME_CHALLENGE_BYTES as usize];
                rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut challenge);
                stream.write_all(&challenge).await.unwrap();
                let mut response = [0u8; FRAME_AUTH_TAG_BYTES as usize];
                stream.read_exact(&mut response).await.unwrap();
                let expected = compute_hmac(&authkey, &challenge);
                assert_eq!(&response[..], &expected[..], "handshake mismatch");
                // Hold the connection briefly so the client can send
                // BUS_SUBSCRIBE if it wants. Then close.
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });
        (dir, path, task)
    }

    #[tokio::test]
    async fn connect_completes_handshake_with_correct_authkey() {
        // C4-2b1 bus_client test 1: handshake succeeds end-to-end
        let authkey = b"shared-secret-32-bytes-exactly!!".to_vec();
        let (_dir, path, server_task) = fake_broker_handshake_only(authkey.clone()).await;

        let client = BusClient::connect(&path, &authkey, "unified-spirit").await;
        assert!(
            client.is_ok(),
            "handshake should succeed: {:?}",
            client.err()
        );
        let _ = server_task.await;
    }

    #[tokio::test]
    async fn connect_fails_with_wrong_authkey() {
        // C4-2b1 bus_client test 2: handshake mismatch is detected
        // server-side. The client `connect()` may still return Ok
        // (handshake bytes flow either way), but subsequent reads
        // surface as `Disconnected`. We verify the recv loop reports
        // disconnection within a reasonable window.
        let server_key = b"server-side-authkey-32-bytes-xx!".to_vec();
        let client_key = b"WRONG-side-authkey-32-bytes-xxx!".to_vec();
        let (_dir, path, _server_task) = fake_broker_handshake_only(server_key.clone()).await;

        let client = BusClient::connect(&path, &client_key, "test")
            .await
            .expect("connect succeeds even with wrong key (server detects)");
        // Wait up to 1s for Disconnected
        let result = tokio::time::timeout(Duration::from_secs(1), client.recv()).await;
        assert!(result.is_ok(), "recv should produce a Disconnected event");
        let event = result.unwrap();
        match event {
            Some(InboundEvent::Disconnected { .. }) => {}
            other => panic!("expected Disconnected, got {other:?}"),
        }
    }

    #[test]
    fn extract_payload_returns_binary_blob() {
        // C4-2b1 bus_client test 3: extract_payload helper round-trips
        let envelope = encode_simple(
            "SPHERE_PULSE",
            Some("titan-trinity-rs"),
            Some("all"),
            Some(&[0x01, 0x02, 0x03, 0x04]),
        )
        .unwrap();
        let extracted = extract_payload(&envelope).unwrap();
        assert_eq!(extracted, vec![0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn extract_payload_returns_none_when_payload_field_missing() {
        // C4-2b1 bus_client test 4: BUS_PING-style envelope with no payload
        let envelope = encode_simple("BUS_PING", Some("broker"), None, None).unwrap();
        assert!(extract_payload(&envelope).is_none());
    }

    #[test]
    fn encode_subscribe_payload_round_trips() {
        // C4-2b1 bus_client test 5: subscribe payload encoder produces
        // valid msgpack with a topics array
        let payload =
            encode_subscribe_payload(&["SPHERE_PULSE", "KERNEL_SHUTDOWN_ANNOUNCE"]).unwrap();
        let value: rmpv::Value =
            rmpv::decode::read_value(&mut std::io::Cursor::new(&payload[..])).unwrap();
        if let rmpv::Value::Map(entries) = value {
            let mut found = false;
            for (k, v) in entries {
                if let rmpv::Value::String(s) = &k {
                    if s.as_str() == Some("topics") {
                        if let rmpv::Value::Array(topics) = v {
                            assert_eq!(topics.len(), 2);
                            found = true;
                        }
                    }
                }
            }
            assert!(found, "topics key present");
        } else {
            panic!("expected msgpack map");
        }
    }

    #[tokio::test]
    async fn subscribe_writes_a_frame_to_broker() {
        // C4-2b1 bus_client test 6: subscribe sends a length-prefixed
        // BUS_SUBSCRIBE envelope to the broker
        let authkey = b"shared-secret-32-bytes-exactly!!".to_vec();
        let dir = tempdir().unwrap();
        let path = dir.path().join("bus.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();

        let received = Arc::new(tokio::sync::Mutex::new(None));
        let received_clone = received.clone();
        let authkey_srv = authkey.clone();
        let server_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            // Handshake
            let mut challenge = [0u8; FRAME_CHALLENGE_BYTES as usize];
            rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut challenge);
            stream.write_all(&challenge).await.unwrap();
            let mut response = [0u8; FRAME_AUTH_TAG_BYTES as usize];
            stream.read_exact(&mut response).await.unwrap();
            let expected = compute_hmac(&authkey_srv, &challenge);
            assert_eq!(&response[..], &expected[..]);

            // Read one length-prefixed frame
            let mut prefix = [0u8; 4];
            stream.read_exact(&mut prefix).await.unwrap();
            let n = u32::from_le_bytes(prefix) as usize;
            let mut payload = vec![0_u8; n];
            stream.read_exact(&mut payload).await.unwrap();
            let mut lock = received_clone.lock().await;
            *lock = Some(payload);
        });

        let client = BusClient::connect(&path, &authkey, "unified-spirit")
            .await
            .unwrap();
        client
            .subscribe(&["SPHERE_PULSE", "KERNEL_SHUTDOWN_ANNOUNCE"])
            .await
            .unwrap();
        // Give server time to read
        tokio::time::sleep(Duration::from_millis(100)).await;
        client.shutdown().await;

        let _ = server_task.await;
        let received = received.lock().await;
        assert!(
            received.is_some(),
            "broker received the BUS_SUBSCRIBE frame"
        );
        let bytes = received.as_ref().unwrap();
        let header = decode_header(bytes).unwrap();
        assert_eq!(header.msg_type.as_deref(), Some("BUS_SUBSCRIBE"));
        assert_eq!(header.src.as_deref(), Some("unified-spirit"));
    }
}
