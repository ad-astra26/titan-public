//! Integration tests for the full broker — real Unix socket, real
//! handshake, real frame round-trip, real fanout.
//!
//! Per PLAN_microkernel_phase_c_s2_kernel.md §13.1 chunk C2-2.b: prove the
//! broker works end-to-end before declaring C-S2.b shipped.

use std::path::Path;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

use titan_bus::BusBroker;
use titan_core::frame::{compute_hmac, decode_length_prefix, encode_frame};

const AUTHKEY: &[u8] = b"shared-broker-authkey-32-bytesXX";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(2);

/// Connect a client to the broker socket + perform handshake. Returns the
/// connected stream.
async fn connect_and_handshake(sock_path: &Path, authkey: &[u8]) -> UnixStream {
    let mut stream = tokio::time::timeout(HANDSHAKE_TIMEOUT, UnixStream::connect(sock_path))
        .await
        .expect("connect timeout")
        .expect("connect failed");

    // 1. Receive challenge (32 raw bytes)
    let mut challenge = [0u8; 32];
    stream.read_exact(&mut challenge).await.unwrap();

    // 2. Send HMAC response (32 raw bytes)
    let response = compute_hmac(authkey, &challenge);
    stream.write_all(&response).await.unwrap();

    stream
}

/// Send a length-prefixed frame.
async fn send_frame(stream: &mut UnixStream, payload: &[u8]) {
    let bytes = encode_frame(payload).unwrap();
    stream.write_all(&bytes).await.unwrap();
}

/// Receive a length-prefixed frame.
async fn recv_frame(stream: &mut UnixStream) -> Vec<u8> {
    let mut prefix = [0u8; 4];
    stream.read_exact(&mut prefix).await.unwrap();
    let n = decode_length_prefix(&prefix).unwrap() as usize;
    if n == 0 {
        return Vec::new();
    }
    let mut payload = vec![0u8; n];
    stream.read_exact(&mut payload).await.unwrap();
    payload
}

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

#[tokio::test]
async fn broker_starts_clean_and_stops_clean() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;
    assert!(sock_path.exists());
    assert_eq!(broker.subscriber_count().await, 0);
    broker.stop().await;
    assert!(!sock_path.exists());
}

#[tokio::test]
async fn client_handshake_succeeds_with_correct_authkey() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let _stream = connect_and_handshake(&sock_path, AUTHKEY).await;

    // Wait briefly for broker to register the subscriber
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(broker.subscriber_count().await, 1);

    drop(_stream);
    tokio::time::sleep(Duration::from_millis(100)).await;
    broker.stop().await;
}

#[tokio::test]
async fn client_handshake_rejected_with_wrong_authkey() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut stream = UnixStream::connect(&sock_path).await.unwrap();
    let mut challenge = [0u8; 32];
    stream.read_exact(&mut challenge).await.unwrap();

    // Send WRONG response
    let bad_authkey = b"WRONG-authkey-32-bytes-exactlyAB";
    let response = compute_hmac(bad_authkey, &challenge);
    stream.write_all(&response).await.unwrap();

    // Broker should close — next read returns EOF (0 bytes)
    let mut buf = [0u8; 1];
    let read_result = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf)).await;
    match read_result {
        Ok(Ok(0)) => {}  // Clean EOF — broker closed
        Ok(Err(_)) => {} // Connection reset — also acceptable
        other => panic!("expected EOF/error, got {other:?}"),
    }

    // Broker subscriber count should NOT have grown
    assert_eq!(broker.subscriber_count().await, 0);
    broker.stop().await;
}

#[tokio::test]
async fn two_clients_publish_fanout() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut alice = connect_and_handshake(&sock_path, AUTHKEY).await;
    let mut bob = connect_and_handshake(&sock_path, AUTHKEY).await;

    // Alice subscribes (sends BUS_SUBSCRIBE with src=alice)
    let alice_sub =
        titan_bus::message::encode_simple("BUS_SUBSCRIBE", Some("alice"), Some("all"), None)
            .unwrap();
    send_frame(&mut alice, &alice_sub).await;

    // Bob subscribes
    let bob_sub =
        titan_bus::message::encode_simple("BUS_SUBSCRIBE", Some("bob"), Some("all"), None).unwrap();
    send_frame(&mut bob, &bob_sub).await;

    // Wait for broker to register both subscribers under their real names
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(broker.subscriber_count().await, 2);

    // Alice publishes a BODY_STATE message; Bob should receive it
    let body_msg = titan_bus::message::encode_simple(
        "BODY_STATE",
        Some("alice"),
        Some("all"),
        Some(b"alice-payload"),
    )
    .unwrap();
    send_frame(&mut alice, &body_msg).await;

    // Bob receives the fanout (alice's publish)
    let received = tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut bob))
        .await
        .expect("bob recv timeout");
    // Verify Bob received Alice's bytes byte-identical
    assert_eq!(received, body_msg);

    // Alice should NOT echo back to herself — verify by setting a short read
    // timeout; alice has no expected messages (no other publishers).
    let alice_read = tokio::time::timeout(Duration::from_millis(200), recv_frame(&mut alice)).await;
    assert!(
        alice_read.is_err(),
        "publisher should not receive its own message"
    );

    broker.stop().await;
}

#[tokio::test]
async fn drift_bridge_dual_emit() {
    // Per SPEC §3.1 D15: publisher sends EPOCH_TICK; subscriber listening on
    // KERNEL_EPOCH_TICK should receive it (and vice versa).
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut publisher = connect_and_handshake(&sock_path, AUTHKEY).await;
    let mut subscriber = connect_and_handshake(&sock_path, AUTHKEY).await;

    let pub_sub =
        titan_bus::message::encode_simple("BUS_SUBSCRIBE", Some("publisher"), Some("all"), None)
            .unwrap();
    send_frame(&mut publisher, &pub_sub).await;
    let sub_sub =
        titan_bus::message::encode_simple("BUS_SUBSCRIBE", Some("subscriber"), Some("all"), None)
            .unwrap();
    send_frame(&mut subscriber, &sub_sub).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Publisher sends LEGACY name "EPOCH_TICK"
    let legacy_msg =
        titan_bus::message::encode_simple("EPOCH_TICK", Some("publisher"), Some("all"), None)
            .unwrap();
    send_frame(&mut publisher, &legacy_msg).await;

    // Subscriber should receive 2 frames (dual-emit: EPOCH_TICK + KERNEL_EPOCH_TICK)
    let frame1 = tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut subscriber))
        .await
        .expect("first frame timeout");
    let frame2 = tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut subscriber))
        .await
        .expect("second frame timeout");

    let hdr1 = titan_bus::message::decode_header(&frame1).unwrap();
    let hdr2 = titan_bus::message::decode_header(&frame2).unwrap();
    let types: std::collections::HashSet<_> = [
        hdr1.msg_type.unwrap_or_default(),
        hdr2.msg_type.unwrap_or_default(),
    ]
    .into_iter()
    .collect();
    assert!(
        types.contains("EPOCH_TICK") && types.contains("KERNEL_EPOCH_TICK"),
        "expected drift-bridge dual-emit; got types: {types:?}"
    );

    broker.stop().await;
}
