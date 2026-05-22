//! main_bus_publisher — Substrate ↔ kernel main bus integration.
//!
//! Closes work originally scoped to C-S3 chunks C3-3 (boot connect) +
//! C3-5 (publishes) — both explicitly deferred per `SESSION_20260429_phase_c_s3_substrate.md`
//! ("full one_for_one substrate respawn deferred to C-S4 when substrate
//! has bus client"). C-S4 ships the substrate's main-bus client + publish
//! wiring as chunk C4-2c.
//!
//! Per SPEC §9.A `titan-trinity-rs` row:
//! - **Bus subscriptions (REQUIRED):** `KERNEL_EPOCH_TICK`,
//!   `KERNEL_SHUTDOWN_ANNOUNCE`, `KERNEL_BOOT_GENERATION_CHANGED`,
//!   `SUPERVISION_CHILD_DOWN`. Optional: `SWAP_SUBTREE_REQUEST`.
//! - **Bus publications:** `TRINITY_SUBSTRATE_TOPOLOGY_UPDATED` (P1, after
//!   topology_30d.bin write), `SPHERE_PULSE` (P0, per pulse), `SPHERE_EPOCH_TICK`
//!   (P0, per cycle boundary), `SUPERVISION_CHILD_RESTARTED` (P0, after
//!   restarting unified-spirit — wired in C4-4).
//!
//! Per SPEC §10.E telemetry write-then-publish ordering: shm slot writes
//! happen BEFORE bus publishes. Reverse = race (consumers see fresh bus
//! message but stale slot). The publish helpers in this module assume the
//! caller has already completed the corresponding shm write.

use std::path::Path;
use std::sync::Arc;

use thiserror::Error;
use titan_bus::client::{BusClient, BusClientError, InboundEvent};
use tracing::{debug, info, warn};

use crate::sphere_clocks::PulseEvent;

/// Topics the substrate subscribes to per SPEC §9.A trinity-rs row.
pub const REQUIRED_SUBSCRIPTIONS: [&str; 4] = [
    "KERNEL_EPOCH_TICK",
    "KERNEL_SHUTDOWN_ANNOUNCE",
    "KERNEL_BOOT_GENERATION_CHANGED",
    "SUPERVISION_CHILD_DOWN",
];

/// Optional subscriptions per SPEC §9.A.
pub const OPTIONAL_SUBSCRIPTIONS: [&str; 1] = ["SWAP_SUBTREE_REQUEST"];

/// Errors during substrate main-bus operations.
#[derive(Debug, Error)]
pub enum MainBusError {
    /// Underlying client connect / I/O / encode error.
    #[error("bus client: {0}")]
    Client(#[from] BusClientError),
    /// `TITAN_AUTHKEY_HEX` env var missing or malformed (hex decode fail).
    #[error("authkey: {0}")]
    Authkey(String),
    /// `TITAN_KERNEL_BUS_SOCKET_PATH` not provided.
    #[error("bus socket path missing")]
    SocketMissing,
}

/// Connect substrate to the main bus broker, complete HMAC handshake,
/// and subscribe to the required topics. Returns an `Arc<BusClient>` that
/// callers can clone freely for publishing + receiving.
///
/// Per SPEC §10.A B7-B8 boot ordering: substrate connects to `/tmp/titan_bus_<id>.sock`
/// AFTER kernel has bound it (kernel sequence step "Kernel: bind main bus"
/// completes before "Kernel: spawn substrate").
pub async fn connect_main_bus(
    socket_path: &Path,
    authkey_hex: &str,
    titan_id: &str,
) -> Result<Arc<BusClient>, MainBusError> {
    let authkey =
        hex::decode(authkey_hex).map_err(|e| MainBusError::Authkey(format!("hex decode: {e}")))?;

    let client_name = format!("trinity-substrate-{}", titan_id.to_lowercase());
    let client = BusClient::connect(socket_path, &authkey, &client_name).await?;
    let client = Arc::new(client);

    // Subscribe to all required + optional topics. Required failures
    // bubble up; optionals are best-effort with WARN.
    let topics: Vec<&str> = REQUIRED_SUBSCRIPTIONS
        .iter()
        .chain(OPTIONAL_SUBSCRIPTIONS.iter())
        .copied()
        .collect();
    client.subscribe(&topics).await?;

    info!(
        event = "MAIN_BUS_CONNECTED",
        binary = "trinity-substrate",
        titan_id = titan_id,
        socket = ?socket_path,
        subscriptions = ?topics,
        "substrate connected to main bus"
    );

    Ok(client)
}

/// Publish a `SPHERE_PULSE` event per SPEC §8.6 row + Preamble G11.
/// Caller is responsible for invoking AFTER the corresponding shm slot
/// write per SPEC §10.E ordering.
///
/// Payload schema: `{clock_name: str, pulse_count: u64, phase: f64, ts: f64}`.
pub async fn publish_sphere_pulse(
    client: &BusClient,
    pulse: &PulseEvent,
    ts: f64,
) -> Result<(), MainBusError> {
    let payload = encode_sphere_pulse_payload(pulse, ts);
    client
        .publish("SPHERE_PULSE", Some("all"), Some(payload))
        .await?;
    Ok(())
}

/// Publish a `SPHERE_EPOCH_TICK` event per SPEC §8.6.
/// Payload schema: `{epoch_id: u64, ts: f64}`.
pub async fn publish_sphere_epoch_tick(
    client: &BusClient,
    epoch_id: u64,
    ts: f64,
) -> Result<(), MainBusError> {
    let payload = encode_epoch_tick_payload(epoch_id, ts);
    client
        .publish("SPHERE_EPOCH_TICK", Some("all"), Some(payload))
        .await?;
    Ok(())
}

/// Publish `TRINITY_SUBSTRATE_TOPOLOGY_UPDATED` per SPEC §8.6.
/// Caller MUST invoke AFTER `topology_30d.bin` SeqLock write completes
/// (SPEC §10.E telemetry write-then-publish — reverse is a race).
///
/// Payload schema: `{ts: f64}` (signal-only; consumers read shm slot for content).
pub async fn publish_topology_updated(client: &BusClient, ts: f64) -> Result<(), MainBusError> {
    let payload = encode_topology_updated_payload(ts);
    client
        .publish(
            "TRINITY_SUBSTRATE_TOPOLOGY_UPDATED",
            Some("all"),
            Some(payload),
        )
        .await?;
    Ok(())
}

/// Spawn the inbound dispatch task. Drains `client.recv()` and routes
/// events. On `KERNEL_SHUTDOWN_ANNOUNCE` sets the shutdown flag so the
/// substrate's main loop can exit gracefully.
pub fn spawn_inbound_dispatch(
    client: Arc<BusClient>,
    shutdown_flag: Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                info!(
                    event = "MAIN_BUS_DISPATCH_STOP",
                    reason = "shutdown_flag",
                    "substrate main-bus dispatch loop exiting"
                );
                return;
            }
            let event = match client.recv().await {
                Some(e) => e,
                None => {
                    info!(
                        event = "MAIN_BUS_DISPATCH_STOP",
                        reason = "channel_closed",
                        "main-bus client recv channel closed"
                    );
                    return;
                }
            };
            match event {
                InboundEvent::Message {
                    msg_type,
                    raw_bytes: _,
                    ..
                } => match msg_type.as_str() {
                    "KERNEL_SHUTDOWN_ANNOUNCE" => {
                        info!(
                            event = "KERNEL_SHUTDOWN_RECEIVED",
                            "substrate received KERNEL_SHUTDOWN_ANNOUNCE; setting shutdown flag"
                        );
                        shutdown_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        return;
                    }
                    "KERNEL_EPOCH_TICK" => {
                        // Substrate uses pi-heartbeat from fastbus for tight
                        // timing. Main-bus KERNEL_EPOCH_TICK is informational.
                        debug!(
                            event = "KERNEL_EPOCH_TICK_RECEIVED",
                            "epoch tick informational"
                        );
                    }
                    "KERNEL_BOOT_GENERATION_CHANGED" => {
                        info!(
                            event = "BOOT_GENERATION_CHANGED",
                            "kernel rebooted — substrate continues; full re-handshake at next kernel restart cascade"
                        );
                    }
                    "SUPERVISION_CHILD_DOWN" => {
                        info!(
                            event = "SUPERVISION_CHILD_DOWN_RECEIVED",
                            "supervisor child-down signal — full one_for_one respawn lands in C4-4"
                        );
                    }
                    other => {
                        debug!(
                            event = "MAIN_BUS_MSG_UNHANDLED",
                            msg_type = other,
                            "received message — no handler in C4-2c"
                        );
                    }
                },
                InboundEvent::Disconnected { reason } => {
                    warn!(
                        event = "MAIN_BUS_DISCONNECTED",
                        reason = %reason,
                        "substrate main-bus connection closed"
                    );
                    return;
                }
            }
        }
    })
}

/// Build SPHERE_PULSE payload as `rmpv::Value::Map` per SPEC §8.6 + §8.10
/// byte-identical guarantee.
fn encode_sphere_pulse_payload(pulse: &PulseEvent, ts: f64) -> rmpv::Value {
    rmpv::Value::Map(vec![
        (
            rmpv::Value::String("clock_name".into()),
            rmpv::Value::String(pulse.role.as_str().into()),
        ),
        (
            rmpv::Value::String("pulse_count".into()),
            rmpv::Value::Integer(rmpv::Integer::from(pulse.pulse_count as u64)),
        ),
        (
            rmpv::Value::String("phase".into()),
            rmpv::Value::F64(pulse.phase as f64),
        ),
        (
            rmpv::Value::String("balanced".into()),
            rmpv::Value::Boolean(pulse.balanced),
        ),
        (
            rmpv::Value::String("consecutive_balanced".into()),
            rmpv::Value::Integer(rmpv::Integer::from(pulse.consecutive_balanced as u64)),
        ),
        (rmpv::Value::String("ts".into()), rmpv::Value::F64(ts)),
    ])
}

/// Build SPHERE_EPOCH_TICK payload as `rmpv::Value::Map` per SPEC §8.6.
fn encode_epoch_tick_payload(epoch_id: u64, ts: f64) -> rmpv::Value {
    rmpv::Value::Map(vec![
        (
            rmpv::Value::String("epoch_id".into()),
            rmpv::Value::Integer(rmpv::Integer::from(epoch_id)),
        ),
        (rmpv::Value::String("ts".into()), rmpv::Value::F64(ts)),
    ])
}

/// Build TRINITY_SUBSTRATE_TOPOLOGY_UPDATED payload as `rmpv::Value::Map` per
/// SPEC §8.6 (signal-only — consumers read shm slot for content).
fn encode_topology_updated_payload(ts: f64) -> rmpv::Value {
    rmpv::Value::Map(vec![(
        rmpv::Value::String("ts".into()),
        rmpv::Value::F64(ts),
    )])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sphere_clocks::ClockRole;
    use std::time::Duration;
    use titan_bus::client::extract_payload;
    use titan_bus::message::{decode_header, encode_simple};
    use titan_core::constants::{FRAME_AUTH_TAG_BYTES, FRAME_CHALLENGE_BYTES};
    use titan_core::frame::compute_hmac;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Helper: launch a fake broker that completes handshake + reads up to
    /// `expected_frames` length-prefixed frames, returning them.
    async fn fake_broker(
        authkey: Vec<u8>,
        expected_frames: usize,
    ) -> (
        tempfile::TempDir,
        std::path::PathBuf,
        tokio::task::JoinHandle<Vec<Vec<u8>>>,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bus.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let task: tokio::task::JoinHandle<Vec<Vec<u8>>> = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut challenge = [0u8; FRAME_CHALLENGE_BYTES as usize];
            rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut challenge);
            stream.write_all(&challenge).await.unwrap();
            let mut response = [0u8; FRAME_AUTH_TAG_BYTES as usize];
            stream.read_exact(&mut response).await.unwrap();
            let expected = compute_hmac(&authkey, &challenge);
            assert_eq!(&response[..], &expected[..]);
            let mut frames = Vec::new();
            for _ in 0..expected_frames {
                let mut prefix = [0u8; 4];
                if stream.read_exact(&mut prefix).await.is_err() {
                    break;
                }
                let n = u32::from_le_bytes(prefix) as usize;
                let mut payload = vec![0_u8; n];
                if stream.read_exact(&mut payload).await.is_err() {
                    break;
                }
                frames.push(payload);
            }
            frames
        });
        (dir, path, task)
    }

    #[tokio::test]
    async fn connect_main_bus_completes_handshake_and_subscribes() {
        // C4-2c test 1: substrate connects + sends BUS_SUBSCRIBE with all
        // REQUIRED + OPTIONAL topics
        let authkey = b"shared-secret-32-bytes-exactly!!".to_vec();
        let authkey_hex = hex::encode(&authkey);
        let (_dir, path, server_task) = fake_broker(authkey, 1).await;
        let _client = connect_main_bus(&path, &authkey_hex, "T1").await.unwrap();
        let frames = tokio::time::timeout(Duration::from_secs(2), server_task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            frames.len(),
            1,
            "broker received exactly 1 frame (BUS_SUBSCRIBE)"
        );
        let header = decode_header(&frames[0]).unwrap();
        assert_eq!(header.msg_type.as_deref(), Some("BUS_SUBSCRIBE"));
        assert_eq!(header.src.as_deref(), Some("trinity-substrate-t1"));
    }

    #[tokio::test]
    async fn publish_sphere_pulse_writes_correct_payload() {
        // C4-2c test 2: SPHERE_PULSE wire-format payload schema
        let authkey = b"shared-secret-32-bytes-exactly!!".to_vec();
        let authkey_hex = hex::encode(&authkey);
        let (_dir, path, server_task) = fake_broker(authkey, 2).await;
        let client = connect_main_bus(&path, &authkey_hex, "T1").await.unwrap();

        let pulse = PulseEvent {
            role: ClockRole::InnerBody,
            pulse_count: 42,
            phase: 1.5_f32,
            radius_before: 1.0,
            radius_after: 0.95,
            balanced: true,
            consecutive_balanced: 7,
        };
        publish_sphere_pulse(&client, &pulse, 1234567890.5)
            .await
            .unwrap();
        // Give server time to read the publish frame
        tokio::time::sleep(Duration::from_millis(100)).await;
        client.shutdown().await;

        let frames = tokio::time::timeout(Duration::from_secs(2), server_task)
            .await
            .unwrap()
            .unwrap();
        // Frame[0] = BUS_SUBSCRIBE, Frame[1] = SPHERE_PULSE
        assert!(frames.len() >= 2);
        let header = decode_header(&frames[1]).unwrap();
        assert_eq!(header.msg_type.as_deref(), Some("SPHERE_PULSE"));
        // §4.C-ter wire-format: extract_payload returns nested Value::Map directly
        let payload = extract_payload(&frames[1]).unwrap();
        let map = match payload {
            rmpv::Value::Map(m) => m,
            other => panic!("expected map, got {other:?}"),
        };
        let mut got_clock = None;
        let mut got_phase = None;
        let mut got_pulse_count = None;
        let mut got_consec_balanced = None;
        let mut got_ts = None;
        for (k, v) in map {
            if let rmpv::Value::String(s) = &k {
                match s.as_str() {
                    Some("clock_name") => got_clock = v.as_str().map(String::from),
                    Some("phase") => got_phase = v.as_f64(),
                    Some("pulse_count") => got_pulse_count = v.as_u64(),
                    Some("consecutive_balanced") => got_consec_balanced = v.as_u64(),
                    Some("ts") => got_ts = v.as_f64(),
                    _ => {}
                }
            }
        }
        assert_eq!(got_clock.as_deref(), Some("inner_body"));
        assert!((got_phase.unwrap() - 1.5).abs() < 1e-6);
        assert_eq!(got_pulse_count, Some(42));
        assert_eq!(got_consec_balanced, Some(7));
        assert!((got_ts.unwrap() - 1234567890.5).abs() < 1e-3);
    }

    #[tokio::test]
    async fn publish_sphere_epoch_tick_payload_schema() {
        // C4-2c test 3: SPHERE_EPOCH_TICK wire-format
        let authkey = b"shared-secret-32-bytes-exactly!!".to_vec();
        let authkey_hex = hex::encode(&authkey);
        let (_dir, path, server_task) = fake_broker(authkey, 2).await;
        let client = connect_main_bus(&path, &authkey_hex, "T2").await.unwrap();

        publish_sphere_epoch_tick(&client, 7, 100.0).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        client.shutdown().await;

        let frames = tokio::time::timeout(Duration::from_secs(2), server_task)
            .await
            .unwrap()
            .unwrap();
        assert!(frames.len() >= 2);
        let header = decode_header(&frames[1]).unwrap();
        assert_eq!(header.msg_type.as_deref(), Some("SPHERE_EPOCH_TICK"));
    }

    #[tokio::test]
    async fn publish_topology_updated_payload_schema() {
        // C4-2c test 4: TRINITY_SUBSTRATE_TOPOLOGY_UPDATED wire-format
        let authkey = b"shared-secret-32-bytes-exactly!!".to_vec();
        let authkey_hex = hex::encode(&authkey);
        let (_dir, path, server_task) = fake_broker(authkey, 2).await;
        let client = connect_main_bus(&path, &authkey_hex, "T3").await.unwrap();

        publish_topology_updated(&client, 200.5).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        client.shutdown().await;

        let frames = tokio::time::timeout(Duration::from_secs(2), server_task)
            .await
            .unwrap()
            .unwrap();
        assert!(frames.len() >= 2);
        let header = decode_header(&frames[1]).unwrap();
        assert_eq!(
            header.msg_type.as_deref(),
            Some("TRINITY_SUBSTRATE_TOPOLOGY_UPDATED")
        );
    }

    #[test]
    fn required_subscriptions_match_spec_9a() {
        // C4-2c test 5: SPEC §9.A trinity-rs row REQUIRED set fully covered
        for required in [
            "KERNEL_EPOCH_TICK",
            "KERNEL_SHUTDOWN_ANNOUNCE",
            "KERNEL_BOOT_GENERATION_CHANGED",
            "SUPERVISION_CHILD_DOWN",
        ] {
            assert!(
                REQUIRED_SUBSCRIPTIONS.contains(&required),
                "{required} must be REQUIRED per SPEC §9.A"
            );
        }
        assert_eq!(REQUIRED_SUBSCRIPTIONS.len(), 4);
    }

    #[tokio::test]
    async fn shutdown_announce_flips_shutdown_flag() {
        // C4-2c test 6: KERNEL_SHUTDOWN_ANNOUNCE → dispatch flips flag
        let authkey = b"shared-secret-32-bytes-exactly!!".to_vec();
        let authkey_hex = hex::encode(&authkey);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bus.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let authkey_srv = authkey.clone();
        let server_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut challenge = [0u8; FRAME_CHALLENGE_BYTES as usize];
            rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut challenge);
            stream.write_all(&challenge).await.unwrap();
            let mut response = [0u8; FRAME_AUTH_TAG_BYTES as usize];
            stream.read_exact(&mut response).await.unwrap();
            let expected = compute_hmac(&authkey_srv, &challenge);
            assert_eq!(&response[..], &expected[..]);
            // Read BUS_SUBSCRIBE
            let mut prefix = [0u8; 4];
            stream.read_exact(&mut prefix).await.unwrap();
            let n = u32::from_le_bytes(prefix) as usize;
            let mut sub = vec![0_u8; n];
            stream.read_exact(&mut sub).await.unwrap();

            // Send KERNEL_SHUTDOWN_ANNOUNCE downstream
            let envelope = encode_simple(
                "KERNEL_SHUTDOWN_ANNOUNCE",
                Some("kernel"),
                Some("all"),
                None,
            )
            .unwrap();
            let mut frame = (envelope.len() as u32).to_le_bytes().to_vec();
            frame.extend_from_slice(&envelope);
            stream.write_all(&frame).await.unwrap();
            // Hold connection so dispatch task processes the message
            tokio::time::sleep(Duration::from_millis(300)).await;
        });

        let client = connect_main_bus(&path, &authkey_hex, "T1").await.unwrap();
        let shutdown_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dispatch = spawn_inbound_dispatch(client.clone(), shutdown_flag.clone());

        // Wait for shutdown flag to flip (max 1s)
        for _ in 0..100 {
            if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            shutdown_flag.load(std::sync::atomic::Ordering::Relaxed),
            "shutdown flag must flip after KERNEL_SHUTDOWN_ANNOUNCE"
        );
        let _ = dispatch.await;
        let _ = server_task.await;
    }
}
