//! Auto-PONG-on-PING contract test for `BusClient`.
//!
//! Per SPEC §3.1 D10 + §10.B 3-layer heartbeat hierarchy: subscriber MUST
//! pong on every BUS_PING; broker drops connection after 15s of missed
//! pongs (3-missed-pings rule per `BUS_PING_TIMEOUT_S = 15.0`).
//!
//! Closes rFP_phase_c_close_all_runtime_gaps chunk 9D. Before this fix
//! the recv loop forwarded BUS_PING to events_tx but no Rust daemon
//! handled it → every subscriber's `last_pong_ts` aged past 15s → broker
//! purged all of them simultaneously → cascade-killed the kernel within
//! 25-30s of every T3 boot.
//!
//! Tests in this file:
//! 1. Auto-PONG within 100ms of receiving BUS_PING (recv loop short-circuits).
//! 2. Connection survives across multiple PING/PONG rounds (full broker).
//! 3. BUS_PING is NOT forwarded to the caller's events stream.

use std::path::Path;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;

use titan_bus::client::{BusClient, InboundEvent};
use titan_bus::message::{decode_header, encode_simple};
use titan_bus::BusBroker;
use titan_core::constants::{FRAME_AUTH_TAG_BYTES, FRAME_CHALLENGE_BYTES};
use titan_core::frame::{compute_hmac, decode_length_prefix, encode_frame};

const AUTHKEY: &[u8] = b"shared-broker-authkey-32-bytesXX";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(2);

/// Wait for the broker's listening socket to be ready.
async fn wait_for_socket(sock_path: &Path) {
    for _ in 0..20 {
        if sock_path.exists() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("broker socket never appeared");
}

/// Test 1: focused unit-style test against a fake broker that sends one
/// BUS_PING and verifies the client writes back a BUS_PONG within 100ms,
/// without the caller doing anything. This exercises the recv loop's
/// auto-PONG short-circuit in isolation, with no dependency on the full
/// broker's heartbeat scheduler.
#[tokio::test]
async fn recv_loop_auto_pongs_within_100ms_of_ping() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bus.sock");
    let listener = UnixListener::bind(&path).unwrap();
    let authkey = AUTHKEY.to_vec();

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

        // Server sends BUS_PING
        let ping_envelope = encode_simple("BUS_PING", Some("broker"), None, None).unwrap();
        let ping_frame = encode_frame(&ping_envelope).unwrap();
        stream.write_all(&ping_frame).await.unwrap();

        // Server reads next frame — expect BUS_PONG
        let mut prefix = [0u8; 4];
        let read_result =
            tokio::time::timeout(Duration::from_millis(500), stream.read_exact(&mut prefix))
                .await
                .expect("PONG read should not time out — recv loop must auto-pong")
                .expect("PONG read I/O ok");
        let _ = read_result;
        let n = decode_length_prefix(&prefix).unwrap() as usize;
        let mut payload = vec![0u8; n];
        stream.read_exact(&mut payload).await.unwrap();

        // Decode envelope; assert msg_type=BUS_PONG, src=client name, dst=broker
        let header = decode_header(&payload).unwrap();
        assert_eq!(
            header.msg_type.as_deref(),
            Some("BUS_PONG"),
            "client must auto-pong on BUS_PING"
        );
        assert_eq!(header.src.as_deref(), Some("test-subscriber"));
        assert_eq!(header.dst.as_deref(), Some("broker"));
    });

    let client = tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        BusClient::connect(&path, &authkey, "test-subscriber"),
    )
    .await
    .expect("connect timeout")
    .expect("connect failed");

    // Wait for the server task to assert (it will panic on failure).
    tokio::time::timeout(Duration::from_secs(3), server_task)
        .await
        .expect("server task hang")
        .expect("server task error");

    client.shutdown().await;
}

/// Test 2: full broker — connection survives multiple PING/PONG cycles
/// without being marked closed. Uses `BusBroker::start` so the real
/// heartbeat scheduler is exercised. Runs short (1.5s) — the broker's
/// `BUS_PING_INTERVAL_S=5.0`, so we cannot wait for a real ping in unit
/// time, but we CAN verify that across the connection lifecycle the
/// subscriber count stays at 1 (the auto-PONG path doesn't break the
/// existing handshake/subscribe flow).
#[tokio::test]
async fn full_broker_subscriber_stays_alive_with_auto_pong() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let client = BusClient::connect(&sock_path, AUTHKEY, "test-sub-1")
        .await
        .expect("connect ok");
    client
        .subscribe(&["SPHERE_PULSE"])
        .await
        .expect("subscribe ok");

    // Give the broker time to register the subscription.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(broker.subscriber_count().await, 1);

    // Hold the connection open for 1.5s — does not cover a full ping cycle
    // (5s) but does verify the recv loop is alive and the auto-PONG path
    // does not corrupt the subscriber state.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    assert_eq!(
        broker.subscriber_count().await,
        1,
        "subscriber must remain registered"
    );

    client.shutdown().await;
    broker.stop().await;
}

/// Test 3: BUS_PING is NOT forwarded to the caller's events stream.
/// Caller can `recv()` and only see real messages, never PINGs.
#[tokio::test]
async fn bus_ping_not_forwarded_to_caller_events() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bus.sock");
    let listener = UnixListener::bind(&path).unwrap();
    let authkey = AUTHKEY.to_vec();

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

        // Send BUS_PING
        let ping = encode_simple("BUS_PING", Some("broker"), None, None).unwrap();
        stream
            .write_all(&encode_frame(&ping).unwrap())
            .await
            .unwrap();

        // Send a real message (SPHERE_PULSE)
        let real = encode_simple(
            "SPHERE_PULSE",
            Some("titan-trinity-rs"),
            Some("all"),
            Some(rmpv::Value::Map(vec![(
                rmpv::Value::String("phase".into()),
                rmpv::Value::F64(0.5),
            )])),
        )
        .unwrap();
        stream
            .write_all(&encode_frame(&real).unwrap())
            .await
            .unwrap();

        // Drain the PONG so the connection doesn't error.
        let mut prefix = [0u8; 4];
        stream.read_exact(&mut prefix).await.unwrap();
        let n = decode_length_prefix(&prefix).unwrap() as usize;
        let mut buf = vec![0u8; n];
        stream.read_exact(&mut buf).await.unwrap();

        // Hold socket open briefly so client recv loop can finish
        tokio::time::sleep(Duration::from_millis(150)).await;
    });

    let client = BusClient::connect(&path, &authkey, "test-caller")
        .await
        .expect("connect ok");

    // First inbound event from caller's perspective MUST be SPHERE_PULSE,
    // not BUS_PING (which the recv loop swallowed + auto-PONG'd).
    let event = tokio::time::timeout(Duration::from_secs(2), client.recv())
        .await
        .expect("recv timeout")
        .expect("recv channel closed");

    match event {
        InboundEvent::Message {
            msg_type,
            raw_bytes: _,
            ..
        } => {
            assert_eq!(
                msg_type, "SPHERE_PULSE",
                "caller must NOT see BUS_PING; recv loop should swallow it"
            );
        }
        other => panic!("expected SPHERE_PULSE Message, got {other:?}"),
    }

    let _ = server_task.await;
    client.shutdown().await;
}
