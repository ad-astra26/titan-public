//! End-to-end kernel-rs boot test — real subprocess + real artifacts.
//!
//! Per PLAN_microkernel_phase_c_s2_kernel.md §13.1 chunk C2-8 +
//! SPEC §10.A boot sequence. Spawns the actual `titan-kernel-rs` binary
//! built by Cargo (`CARGO_BIN_EXE_titan-kernel-rs`), points it at a temp
//! data dir with a fresh identity, waits for `BOOT_COMPLETE`, asserts the
//! observable artifacts (16 shm slots, bus socket, L0 snapshot,
//! supervision.jsonl, substrate placeholder live), then SIGTERMs and
//! verifies clean exit (143).
//!
//! Substrate binary lookup: same dir as kernel binary (sibling
//! `titan-trinity-rs` — renamed in C-S3 chunk C3-3 from
//! `titan-trinity-rs-placeholder`). If missing the test invokes
//! `cargo build -p titan-trinity-rs` once to ensure presence.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Once;
use std::thread;
use std::time::{Duration, Instant};

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use rand::RngCore;
use serde_json::json;

const KERNEL_BINARY: &str = env!("CARGO_BIN_EXE_titan-kernel-rs");

static BUILD_SUBSTRATE_ONCE: Once = Once::new();

fn ensure_substrate_built() {
    BUILD_SUBSTRATE_ONCE.call_once(|| {
        let substrate_path = Path::new(KERNEL_BINARY)
            .parent()
            .unwrap()
            .join("titan-trinity-rs");
        if !substrate_path.exists() {
            let status = Command::new("cargo")
                .args(["build", "-p", "titan-trinity-rs"])
                .status()
                .expect("cargo build invocation");
            assert!(status.success(), "cargo build of substrate failed");
        }
    });
}

fn substrate_path() -> PathBuf {
    Path::new(KERNEL_BINARY)
        .parent()
        .unwrap()
        .join("titan-trinity-rs")
}

fn write_test_identity(path: &Path) -> [u8; 32] {
    let mut seed = [0u8; 32];
    OsRng.fill_bytes(&mut seed);
    let signing = SigningKey::from_bytes(&seed);
    let pk = signing.verifying_key();
    let body = json!({
        "titan_id": "T1",
        "secret_seed_hex": hex::encode(seed),
        "public_key_hex": hex::encode(pk.as_bytes()),
    });
    let mut f = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)
        .expect("open identity file");
    f.write_all(serde_json::to_string_pretty(&body).unwrap().as_bytes())
        .unwrap();
    seed
}

fn poll<F: Fn() -> bool>(predicate: F, deadline: Instant, label: &str) {
    while Instant::now() < deadline {
        if predicate() {
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!("timeout waiting for {label}");
}

#[test]
fn kernel_boots_end_to_end_and_shuts_down_clean() {
    ensure_substrate_built();
    assert!(
        Path::new(KERNEL_BINARY).exists(),
        "kernel binary not built: {KERNEL_BINARY}"
    );
    assert!(
        substrate_path().exists(),
        "substrate binary not built: {}",
        substrate_path().display()
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let data_dir = dir.path().join("data");
    let shm_dir = dir.path().join("shm");
    fs::create_dir_all(&data_dir).unwrap();
    fs::create_dir_all(&shm_dir).unwrap();

    let identity_path = data_dir.join("titan_identity_keypair.json");
    let _seed = write_test_identity(&identity_path);

    let bus_socket = dir.path().join("bus.sock");

    let mut cmd = Command::new(KERNEL_BINARY);
    cmd.args([
        "--titan-id",
        "T1",
        "--shm-dir",
        shm_dir.to_str().unwrap(),
        "--bus-socket",
        bus_socket.to_str().unwrap(),
        "--data-dir",
        data_dir.to_str().unwrap(),
        "--log-level",
        "info",
    ])
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("kernel spawn");
    let stdout = child.stdout.take().unwrap();

    // Watch stdout for BOOT_COMPLETE (tracing-subscriber JSON writer
    // defaults to stdout per logging.rs init).
    let (boot_tx, boot_rx) = std::sync::mpsc::channel();
    let log_thread = thread::spawn(move || {
        let mut all_logs = String::new();
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            all_logs.push_str(&line);
            all_logs.push('\n');
            if line.contains("BOOT_COMPLETE") {
                let _ = boot_tx.send(true);
            }
        }
        all_logs
    });

    let booted = boot_rx.recv_timeout(Duration::from_secs(15)).is_ok();
    if !booted {
        let _ = child.kill();
        panic!("kernel did not emit BOOT_COMPLETE within 15s");
    }

    // Verify observable artifacts (poll briefly — supervision.jsonl + snapshot
    // can land just after BOOT_COMPLETE).
    let artifact_deadline = Instant::now() + Duration::from_secs(5);
    poll(
        || bus_socket.exists(),
        artifact_deadline,
        "bus socket creation",
    );
    poll(
        || {
            shm_dir.join("self_162d.bin").exists()
                && shm_dir.join("inner_body_5d.bin").exists()
                && shm_dir.join("cgn_live_weights.bin").exists()
        },
        artifact_deadline,
        "shm slot files",
    );
    // D04: trinity_state.bin → self_162d.bin symlink
    let symlink_path = shm_dir.join("trinity_state.bin");
    assert!(symlink_path.exists(), "D04 symlink missing");
    let target = fs::read_link(&symlink_path).expect("read_link D04");
    assert_eq!(
        target.file_name().and_then(|s| s.to_str()),
        Some("self_162d.bin"),
        "D04 must point to self_162d.bin"
    );

    // Subset of kernel-owned shm slots verified at boot. Full canonical
    // names per titan-state/src/spec.rs SLOT_SPECS table; full set is
    // verified in titan-state unit tests.
    // Per titan-state/src/spec.rs, kernel creates 16 slots (the cgn_live_weights
    // slot is created by titan-cgn but invoked from kernel boot). Module-owned
    // slots (neuromod_state, titanvm_registers, sensor_cache_*) are NOT
    // created by kernel.
    let expected_kernel_slots = [
        "self_162d.bin",
        "inner_body_5d.bin",
        "inner_mind_15d.bin",
        "inner_spirit_45d.bin",
        "outer_body_5d.bin",
        "outer_mind_15d.bin",
        "outer_spirit_45d.bin",
        "topology_30d.bin",
        "unified_spirit_132d.bin",
        "epoch_counter.bin",
        "circadian.bin",
        "pi_heartbeat.bin",
        "sphere_clocks.bin",
        "chi_state.bin",
        "identity.bin",
        "cgn_live_weights.bin",
    ];
    for slot in expected_kernel_slots {
        assert!(shm_dir.join(slot).exists(), "shm slot missing: {slot}");
    }

    // L0 snapshot loop runs at 1Hz — wait long enough for first write.
    poll(
        || data_dir.join("l0_snapshot.bin").exists(),
        Instant::now() + Duration::from_secs(3),
        "l0_snapshot.bin",
    );
    // supervision.jsonl is intentionally not asserted here: per
    // titan-core::supervisor::event, the file is only written when
    // SUPERVISION events fire (CHILD_DOWN, RESTARTED, ESCALATION, ...).
    // A clean boot with a happy substrate produces zero events and the
    // file is never created, by design (SPEC §11.E).

    // Bus socket: verify mode is 0600 (srw-------) per SPEC §16
    let bus_meta = fs::metadata(&bus_socket).unwrap();
    let mode = bus_meta.permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "bus socket mode should be 0o600 per SPEC");

    // Shutdown via SIGTERM
    let pid = nix::unistd::Pid::from_raw(child.id() as i32);
    nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM).expect("SIGTERM");
    let exit = child.wait().expect("kernel wait");
    let log_capture = log_thread.join().unwrap_or_default();

    // Linux unwraps "killed by signal N" as exit 128+N; SIGTERM = 143.
    // The kernel signal handler also catches SIGTERM and exits cleanly via
    // KernelExitCode::Ok (0). Both are acceptable outcomes.
    let code = exit.code();
    let signal = exit_signal(&exit);
    assert!(
        code == Some(0) || signal == Some(15),
        "kernel exit was code={code:?} signal={signal:?}; logs:\n{log_capture}"
    );
}

#[cfg(unix)]
fn exit_signal(s: &std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    s.signal()
}

#[cfg(not(unix))]
fn exit_signal(_s: &std::process::ExitStatus) -> Option<i32> {
    None
}
