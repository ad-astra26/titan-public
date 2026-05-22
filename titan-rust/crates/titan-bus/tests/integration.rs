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

/// Encode a `BUS_SUBSCRIBE` envelope with `{name, topics, reply_only}`
/// NESTED-MAP payload — mirrors Python BusSocketClient + SPEC §8.2 v1.4.0
/// (`decode_bus_subscribe_payload` expects `payload` to be a Map, not
/// Binary). Used by tests to declare each subscriber's intent.
fn encode_subscribe_with_topics(src: &str, topics: &[&str]) -> Vec<u8> {
    encode_subscribe_with_intent(src, topics, false)
}

/// D-SPEC-42 (SPEC v1.4.0, 2026-05-12) — encode BUS_SUBSCRIBE with
/// explicit `reply_only` intent. Used by tests that verify the broker's
/// silent-skip-on-broadcast behavior for reply-only subscribers.
fn encode_subscribe_with_intent(src: &str, topics: &[&str], reply_only: bool) -> Vec<u8> {
    let topic_values: Vec<rmpv::Value> = topics
        .iter()
        .map(|t| rmpv::Value::String((*t).into()))
        .collect();
    let payload_map = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("name".into()),
            rmpv::Value::String(src.into()),
        ),
        (
            rmpv::Value::String("topics".into()),
            rmpv::Value::Array(topic_values),
        ),
        (
            rmpv::Value::String("reply_only".into()),
            rmpv::Value::Boolean(reply_only),
        ),
    ]);
    let envelope = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("type".into()),
            rmpv::Value::String("BUS_SUBSCRIBE".into()),
        ),
        (
            rmpv::Value::String("src".into()),
            rmpv::Value::String(src.into()),
        ),
        (
            rmpv::Value::String("dst".into()),
            rmpv::Value::String("broker".into()),
        ),
        (rmpv::Value::String("payload".into()), payload_map),
    ]);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &envelope).unwrap();
    out
}
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

    // Alice subscribes with explicit topics — post-§4.C contract: all
    // subscribers MUST declare `broadcast_topics`. Subscribe-all (empty
    // topics) now triggers WARN+drop per rFP_worker_broadcast_topics_completion
    // §4.C.
    let alice_sub = encode_subscribe_with_topics("alice", &["BODY_STATE"]);
    send_frame(&mut alice, &alice_sub).await;

    // Bob subscribes to BODY_STATE
    let bob_sub = encode_subscribe_with_topics("bob", &["BODY_STATE"]);
    send_frame(&mut bob, &bob_sub).await;

    // Wait for broker to register both subscribers under their real names
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(broker.subscriber_count().await, 2);

    // Alice publishes a BODY_STATE message; Bob should receive it
    let body_msg = titan_bus::message::encode_simple(
        "BODY_STATE",
        Some("alice"),
        Some("all"),
        Some(rmpv::Value::String("alice-payload".into())),
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

    // Subscribe with both legacy + canonical drift-bridge names so the
    // dual-emit fanout delivers BOTH frames to subscriber. Per rFP §4.C
    // empty-topics path is now WARN+drop; declare exactly what we expect.
    let pub_sub = encode_subscribe_with_topics("publisher", &["EPOCH_TICK", "KERNEL_EPOCH_TICK"]);
    send_frame(&mut publisher, &pub_sub).await;
    let sub_sub = encode_subscribe_with_topics("subscriber", &["EPOCH_TICK", "KERNEL_EPOCH_TICK"]);
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

/// Post-rFP_worker_broadcast_topics_completion §4.C contract: broker MUST
/// filter `dst="all"` broadcasts by per-subscriber `subscribed_topics`.
/// A subscriber that declares `["BODY_STATE"]` does NOT receive
/// `MIND_STATE` even though both are broadcast `dst=all`.
///
/// This closes the parity gap with `BusSocketServer.publish` lines 798-809
/// + Python side §4.C retirement. Discovered 2026-05-12 via T3 soak gate
/// (816,560 `neuromod_module` queue-full drops in 24h on T3 because the
/// Rust DivineBus didn't honor `broadcast_topics`).
#[tokio::test]
async fn broadcast_filter_blocks_unsubscribed_types() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut publisher = connect_and_handshake(&sock_path, AUTHKEY).await;
    let mut body_listener = connect_and_handshake(&sock_path, AUTHKEY).await;

    // publisher subscribes only because all clients must declare topics
    // (post-§4.C); it doesn't receive its own publishes anyway.
    let pub_sub = encode_subscribe_with_topics("publisher", &["BODY_STATE"]);
    send_frame(&mut publisher, &pub_sub).await;

    // body_listener subscribes to BODY_STATE only — MIND_STATE must be filtered out
    let body_sub = encode_subscribe_with_topics("body_listener", &["BODY_STATE"]);
    send_frame(&mut body_listener, &body_sub).await;

    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(broker.subscriber_count().await, 2);

    // Publish a MIND_STATE — body_listener must NOT receive it
    let mind_msg = titan_bus::message::encode_simple(
        "MIND_STATE",
        Some("publisher"),
        Some("all"),
        Some(rmpv::Value::String("mind-payload".into())),
    )
    .unwrap();
    send_frame(&mut publisher, &mind_msg).await;

    let body_read =
        tokio::time::timeout(Duration::from_millis(300), recv_frame(&mut body_listener)).await;
    assert!(
        body_read.is_err(),
        "body_listener subscribed to BODY_STATE only, but received: {:?}",
        body_read
    );

    // Now publish a BODY_STATE — body_listener MUST receive it
    let body_msg = titan_bus::message::encode_simple(
        "BODY_STATE",
        Some("publisher"),
        Some("all"),
        Some(rmpv::Value::String("body-payload".into())),
    )
    .unwrap();
    send_frame(&mut publisher, &body_msg).await;

    let body_received =
        tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut body_listener))
            .await
            .expect("body_listener never received BODY_STATE despite subscribing");
    let hdr = titan_bus::message::decode_header(&body_received).unwrap();
    assert_eq!(hdr.msg_type.as_deref(), Some("BODY_STATE"));

    broker.stop().await;
}

/// Targeted routing (`dst != "all"`) bypasses the `subscribed_topics`
/// filter — a worker addressed by name MUST receive RPC replies + control
/// messages regardless of its broadcast filter. Mirrors Python lines
/// 794-797 (the `if dst != "all"` branch).
#[tokio::test]
async fn targeted_routing_bypasses_topics_filter() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut publisher = connect_and_handshake(&sock_path, AUTHKEY).await;
    let mut worker = connect_and_handshake(&sock_path, AUTHKEY).await;

    // worker declares topics that do NOT include MODULE_SHUTDOWN
    let pub_sub = encode_subscribe_with_topics("publisher", &["BODY_STATE"]);
    send_frame(&mut publisher, &pub_sub).await;
    let worker_sub = encode_subscribe_with_topics("worker", &["BODY_STATE"]);
    send_frame(&mut worker, &worker_sub).await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Publisher sends MODULE_SHUTDOWN TARGETED to "worker" — must arrive
    // despite worker not declaring MODULE_SHUTDOWN in topics, because
    // targeted routing bypasses the broadcast filter.
    let shutdown_msg = titan_bus::message::encode_simple(
        "MODULE_SHUTDOWN",
        Some("publisher"),
        Some("worker"),
        Some(rmpv::Value::String("shutdown-payload".into())),
    )
    .unwrap();
    send_frame(&mut publisher, &shutdown_msg).await;

    let received = tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut worker))
        .await
        .expect("targeted MODULE_SHUTDOWN was filtered (regression)");
    let hdr = titan_bus::message::decode_header(&received).unwrap();
    assert_eq!(hdr.msg_type.as_deref(), Some("MODULE_SHUTDOWN"));

    broker.stop().await;
}

// ── D-SPEC-42 (SPEC v1.4.0, 2026-05-12) reply_only contract tests ────────────

/// reply_only=true subscribers MUST NOT receive `dst="all"` broadcasts.
/// Mirrors Python `tests/test_bus_socket_broadcast_filter.py
/// ::test_reply_only_subscriber_silently_skipped_on_broadcast`.
#[tokio::test]
async fn reply_only_subscriber_silently_skipped_on_broadcast() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut publisher = connect_and_handshake(&sock_path, AUTHKEY).await;
    let mut rpc_reply_queue = connect_and_handshake(&sock_path, AUTHKEY).await;

    // Publisher is a normal broadcast consumer (declared topics).
    let pub_sub = encode_subscribe_with_topics("publisher", &["BODY_STATE"]);
    send_frame(&mut publisher, &pub_sub).await;

    // rpc_reply_queue is reply_only=true with topics=[] — the canonical
    // pattern for RPC reply queues per SPEC §8.2 v1.4.0 D-SPEC-42.
    let reply_sub = encode_subscribe_with_intent("rpc_reply_queue", &[], true);
    send_frame(&mut rpc_reply_queue, &reply_sub).await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Publisher broadcasts BODY_STATE — reply_only sub MUST NOT receive it.
    let body_msg = titan_bus::message::encode_simple(
        "BODY_STATE",
        Some("publisher"),
        Some("all"),
        Some(rmpv::Value::String("body-payload".into())),
    )
    .unwrap();
    send_frame(&mut publisher, &body_msg).await;

    let reply_read =
        tokio::time::timeout(Duration::from_millis(300), recv_frame(&mut rpc_reply_queue)).await;
    assert!(
        reply_read.is_err(),
        "reply_only=true subscriber received broadcast — D-SPEC-42 contract broken: {:?}",
        reply_read
    );

    broker.stop().await;
}

/// reply_only=true subscriber DOES receive targeted dst=<name> messages.
/// Confirms the SPEC §8.2 v1.4.0 carve-out: only broadcast fan-out is
/// affected; targeted routing is unconditional.
#[tokio::test]
async fn reply_only_subscriber_receives_targeted_messages() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut publisher = connect_and_handshake(&sock_path, AUTHKEY).await;
    let mut rpc_reply_queue = connect_and_handshake(&sock_path, AUTHKEY).await;

    let pub_sub = encode_subscribe_with_topics("publisher", &["BODY_STATE"]);
    send_frame(&mut publisher, &pub_sub).await;
    let reply_sub = encode_subscribe_with_intent("rpc_reply_queue", &[], true);
    send_frame(&mut rpc_reply_queue, &reply_sub).await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Publisher sends targeted RESPONSE — reply_only sub MUST receive it.
    let response_msg = titan_bus::message::encode_simple(
        "RESPONSE",
        Some("publisher"),
        Some("rpc_reply_queue"),
        Some(rmpv::Value::String("result-payload".into())),
    )
    .unwrap();
    send_frame(&mut publisher, &response_msg).await;

    let received = tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut rpc_reply_queue))
        .await
        .expect("reply_only subscriber missed targeted RESPONSE (regression)");
    let hdr = titan_bus::message::decode_header(&received).unwrap();
    assert_eq!(hdr.msg_type.as_deref(), Some("RESPONSE"));

    broker.stop().await;
}

/// Backward compatibility: pre-v1.4.0 BUS_SUBSCRIBE frames omit the
/// `reply_only` key. The broker decoder MUST default to `reply_only=false`
/// (broadcast-consumer intent), preserving byte-identical prior behavior.
#[tokio::test]
async fn missing_reply_only_field_defaults_to_false() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut publisher = connect_and_handshake(&sock_path, AUTHKEY).await;
    let mut legacy_sub = connect_and_handshake(&sock_path, AUTHKEY).await;

    // Build a pre-v1.4.0 BUS_SUBSCRIBE frame (no reply_only key in payload).
    let payload_map = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("name".into()),
            rmpv::Value::String("legacy_sub".into()),
        ),
        (
            rmpv::Value::String("topics".into()),
            rmpv::Value::Array(vec![rmpv::Value::String("BODY_STATE".into())]),
        ),
        // intentionally NO reply_only key — simulates pre-v1.4.0 client
    ]);
    let envelope = rmpv::Value::Map(vec![
        (
            rmpv::Value::String("type".into()),
            rmpv::Value::String("BUS_SUBSCRIBE".into()),
        ),
        (
            rmpv::Value::String("src".into()),
            rmpv::Value::String("legacy_sub".into()),
        ),
        (
            rmpv::Value::String("dst".into()),
            rmpv::Value::String("broker".into()),
        ),
        (rmpv::Value::String("payload".into()), payload_map),
    ]);
    let mut legacy_frame = Vec::new();
    rmpv::encode::write_value(&mut legacy_frame, &envelope).unwrap();
    send_frame(&mut legacy_sub, &legacy_frame).await;

    // publisher subscribes for echo prevention
    let pub_sub = encode_subscribe_with_topics("publisher", &["BODY_STATE"]);
    send_frame(&mut publisher, &pub_sub).await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Broadcast BODY_STATE — legacy_sub MUST receive (reply_only default false).
    let body_msg = titan_bus::message::encode_simple(
        "BODY_STATE",
        Some("publisher"),
        Some("all"),
        Some(rmpv::Value::String("body-payload".into())),
    )
    .unwrap();
    send_frame(&mut publisher, &body_msg).await;

    let received = tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut legacy_sub))
        .await
        .expect("legacy_sub missed broadcast — backward-compat broken");
    let hdr = titan_bus::message::decode_header(&received).unwrap();
    assert_eq!(hdr.msg_type.as_deref(), Some("BODY_STATE"));

    broker.stop().await;
}

// ─────────────────────────────────────────────────────────────────────────
// SPEC §8.0.bis — Boot-window buffer integration tests
// ─────────────────────────────────────────────────────────────────────────
// Closes the bootstrap-race class observed live on T3 2026-05-13
// (agency_worker stuck in state=starting because MODULE_READY emitted
// before Guardian's "guardian" alias subscribed → broker dropped frame).
//
// Per Phase A of rFP_phase_c_bus_delivery_continuity_and_hot_reload.md.

#[tokio::test]
async fn boot_buffer_delivers_module_ready_to_late_attaching_subscriber() {
    // Scenario: live T3 agency_worker bug.
    //   1. Worker connects + subscribes
    //   2. Worker emits MODULE_READY targeting dst="guardian"
    //      — but "guardian" has NOT yet subscribed
    //   3. Pre-fix: frame silently dropped → state stuck "starting"
    //   4. Post-fix (SPEC §8.0.bis): frame buffered in
    //      boot_buffer["guardian"]
    //   5. Guardian-aliased connection attaches later
    //   6. Buffer drains in arrival order → guardian receives MODULE_READY
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut worker = connect_and_handshake(&sock_path, AUTHKEY).await;
    let worker_sub = encode_subscribe_with_topics("test_worker", &["MODULE_HEARTBEAT"]);
    send_frame(&mut worker, &worker_sub).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Emit MODULE_READY targeted dst="guardian" — no guardian subscribed yet.
    let module_ready = titan_bus::message::encode_simple(
        "MODULE_READY",
        Some("test_worker"),
        Some("guardian"),
        Some(rmpv::Value::Map(vec![(
            rmpv::Value::String("src".into()),
            rmpv::Value::String("test_worker".into()),
        )])),
    )
    .unwrap();
    send_frame(&mut worker, &module_ready).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Now guardian subscribes — reply_only=true (it only receives targeted).
    let mut guardian = connect_and_handshake(&sock_path, AUTHKEY).await;
    let guardian_sub = encode_subscribe_with_intent("guardian", &[], /*reply_only=*/ true);
    send_frame(&mut guardian, &guardian_sub).await;

    // Buffer drains on the BUS_SUBSCRIBE processing → guardian receives.
    let received = tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut guardian))
        .await
        .expect("guardian missed buffered MODULE_READY — boot-buffer broken");
    let hdr = titan_bus::message::decode_header(&received).unwrap();
    assert_eq!(
        hdr.msg_type.as_deref(),
        Some("MODULE_READY"),
        "first drained frame must be MODULE_READY"
    );

    broker.stop().await;
}

#[tokio::test]
async fn boot_buffer_drains_multiple_frames_in_arrival_order() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut worker = connect_and_handshake(&sock_path, AUTHKEY).await;
    let worker_sub = encode_subscribe_with_topics("test_worker", &["MODULE_HEARTBEAT"]);
    send_frame(&mut worker, &worker_sub).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Emit 3 heartbeats before guardian attaches.
    for i in 1..=3u8 {
        let hb = titan_bus::message::encode_simple(
            "MODULE_HEARTBEAT",
            Some("test_worker"),
            Some("guardian"),
            Some(rmpv::Value::Integer(rmpv::Integer::from(i))),
        )
        .unwrap();
        send_frame(&mut worker, &hb).await;
    }
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut guardian = connect_and_handshake(&sock_path, AUTHKEY).await;
    let guardian_sub = encode_subscribe_with_intent("guardian", &[], true);
    send_frame(&mut guardian, &guardian_sub).await;

    // Receive 3 frames; assert ordering preserved (msg_type sequence).
    for _ in 0..3 {
        let received = tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut guardian))
            .await
            .expect("guardian missed buffered heartbeat");
        let hdr = titan_bus::message::decode_header(&received).unwrap();
        assert_eq!(hdr.msg_type.as_deref(), Some("MODULE_HEARTBEAT"));
    }

    broker.stop().await;
}

#[tokio::test]
async fn boot_buffer_does_not_buffer_non_listed_types() {
    // Negative: non-BOOT_BUFFERED_TYPES drop on no-subscriber (existing
    // SPEC §8.2 D-SPEC-42 behavior — broker WARN+drop or silent based
    // on subscriber intent; for absent dst, drop). Boot-buffer must
    // NOT absorb application messages.
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut worker = connect_and_handshake(&sock_path, AUTHKEY).await;
    let worker_sub = encode_subscribe_with_topics("test_worker", &["BODY_STATE"]);
    send_frame(&mut worker, &worker_sub).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // BODY_STATE targeted to a name with no subscriber. NOT in
    // BOOT_BUFFERED_TYPES — must drop, not buffer.
    let body_msg = titan_bus::message::encode_simple(
        "BODY_STATE",
        Some("test_worker"),
        Some("late_subscriber"),
        Some(rmpv::Value::String("payload".into())),
    )
    .unwrap();
    send_frame(&mut worker, &body_msg).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut late = connect_and_handshake(&sock_path, AUTHKEY).await;
    let late_sub = encode_subscribe_with_topics("late_subscriber", &["BODY_STATE"]);
    send_frame(&mut late, &late_sub).await;

    // Late MUST NOT receive (non-buffered type dropped at fanout time).
    let result = tokio::time::timeout(Duration::from_millis(500), recv_frame(&mut late)).await;
    assert!(
        result.is_err(),
        "non-boot-buffered type must drop, not buffer (got frame instead)"
    );

    broker.stop().await;
}

// ── D-SPEC-52 (SPEC v1.7.3, 2026-05-14) publisher-skip parity tests ──────────
//
// Per rFP_broker_publisher_skip_parity_fix: the broker's publisher-skip rule
// (preventing self-echo) MUST apply ONLY to dst="all" broadcasts. Targeted
// dst=<name> MUST deliver to the named subscriber even when the publisher IS
// that subscriber (e.g. spirit_worker emits META_CGN_SIGNAL with dst="spirit"
// where MetaCGNConsumer lives in-process).
//
// Pre-D-SPEC-52 the broker.rs fanout filter `map_key.as_str() != from_name`
// was applied UNCONDITIONALLY, dropping these legitimate self-targeted
// deliveries. Symptom: META-CGN + EMOT-CGN signal pipelines silent fleet-wide
// post-fleet-Phase-C migration on 2026-05-14.
//
// Mirrors Python tests/test_broker_publisher_skip_parity.py.

/// Self-targeted dst=<own-name> MUST deliver to publisher per SPEC §8.2
/// v1.4.0 D-SPEC-42 + D-SPEC-52 (v1.7.3). This is the critical bug-catcher
/// for the META-CGN signal-starvation regression.
#[tokio::test]
async fn self_targeted_dst_delivers_to_publisher() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    // Single subscriber acting as both publisher AND destination.
    let mut spirit = connect_and_handshake(&sock_path, AUTHKEY).await;
    let sub = encode_subscribe_with_topics("spirit", &["META_CGN_SIGNAL"]);
    send_frame(&mut spirit, &sub).await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    // spirit publishes a targeted META_CGN_SIGNAL with dst="spirit" —
    // mirrors emit_meta_cgn_signal from inside spirit_worker.
    let msg = titan_bus::message::encode_simple(
        "META_CGN_SIGNAL",
        Some("spirit"),
        Some("spirit"),
        Some(rmpv::Value::String("self-target-payload".into())),
    )
    .unwrap();
    send_frame(&mut spirit, &msg).await;

    let received = tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut spirit))
        .await
        .expect(
            "self-targeted dst=<own-name> was filtered — D-SPEC-52 contract broken; \
             this is the META-CGN signal-starvation regression",
        );
    let hdr = titan_bus::message::decode_header(&received).unwrap();
    assert_eq!(hdr.msg_type.as_deref(), Some("META_CGN_SIGNAL"));
    assert_eq!(hdr.dst.as_deref(), Some("spirit"));

    broker.stop().await;
}

/// Self-broadcast dst="all" MUST skip publisher per SPEC §8.2 v1.4.0 —
/// the broadcast publisher-skip rule (anti-feedback-loop) remains correct.
/// The fix only carved out the targeted case; broadcasts are unchanged.
#[tokio::test]
async fn self_broadcast_dst_all_skips_publisher() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut spirit = connect_and_handshake(&sock_path, AUTHKEY).await;
    let mut cgn = connect_and_handshake(&sock_path, AUTHKEY).await;

    let spirit_sub = encode_subscribe_with_topics("spirit", &["BODY_STATE"]);
    send_frame(&mut spirit, &spirit_sub).await;
    let cgn_sub = encode_subscribe_with_topics("cgn", &["BODY_STATE"]);
    send_frame(&mut cgn, &cgn_sub).await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    // spirit broadcasts BODY_STATE to dst="all".
    let body_msg = titan_bus::message::encode_simple(
        "BODY_STATE",
        Some("spirit"),
        Some("all"),
        Some(rmpv::Value::String("body-payload".into())),
    )
    .unwrap();
    send_frame(&mut spirit, &body_msg).await;

    // cgn MUST receive (non-publisher broadcast subscriber).
    let cgn_received = tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut cgn))
        .await
        .expect("non-publisher broadcast subscriber MUST receive broadcast");
    let cgn_hdr = titan_bus::message::decode_header(&cgn_received).unwrap();
    assert_eq!(cgn_hdr.msg_type.as_deref(), Some("BODY_STATE"));

    // spirit (the publisher) MUST NOT receive its own broadcast.
    let spirit_result =
        tokio::time::timeout(Duration::from_millis(500), recv_frame(&mut spirit)).await;
    assert!(
        spirit_result.is_err(),
        "publisher received its OWN broadcast — anti-feedback-loop broken: {:?}",
        spirit_result
    );

    broker.stop().await;
}

/// Cross-worker targeted dst=<other-name> MUST deliver to the addressed
/// subscriber (sanity check — the bug only affected self-targeted, this
/// path was already correct, but verify no regression).
#[tokio::test]
async fn cross_worker_targeted_delivers_normally_d_spec_51() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("titan_bus_T1.sock");
    let mut broker = BusBroker::new("T1", AUTHKEY.to_vec());
    broker.start(&sock_path).await.unwrap();
    wait_for_socket(&sock_path).await;

    let mut cgn = connect_and_handshake(&sock_path, AUTHKEY).await;
    let mut spirit = connect_and_handshake(&sock_path, AUTHKEY).await;

    let cgn_sub = encode_subscribe_with_topics("cgn", &["META_CGN_SIGNAL"]);
    send_frame(&mut cgn, &cgn_sub).await;
    let spirit_sub = encode_subscribe_with_topics("spirit", &["META_CGN_SIGNAL"]);
    send_frame(&mut spirit, &spirit_sub).await;
    tokio::time::sleep(Duration::from_millis(150)).await;

    // cgn emits META_CGN_SIGNAL with dst="spirit" (P10
    // knowledge.impasse_resolved producer path).
    let msg = titan_bus::message::encode_simple(
        "META_CGN_SIGNAL",
        Some("knowledge"),
        Some("spirit"),
        Some(rmpv::Value::String("cross-worker-payload".into())),
    )
    .unwrap();
    send_frame(&mut cgn, &msg).await;

    // spirit MUST receive.
    let spirit_received = tokio::time::timeout(Duration::from_secs(2), recv_frame(&mut spirit))
        .await
        .expect("cross-worker targeted dst=spirit was filtered");
    let spirit_hdr = titan_bus::message::decode_header(&spirit_received).unwrap();
    assert_eq!(spirit_hdr.msg_type.as_deref(), Some("META_CGN_SIGNAL"));

    // cgn MUST NOT receive (different name + not subscribed via dst).
    let cgn_result = tokio::time::timeout(Duration::from_millis(500), recv_frame(&mut cgn)).await;
    assert!(
        cgn_result.is_err(),
        "targeted dst=spirit leaked to cgn subscriber: {:?}",
        cgn_result
    );

    broker.stop().await;
}
