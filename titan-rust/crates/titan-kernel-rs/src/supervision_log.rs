//! supervision_log — Rotating JSONL writer for `data/supervision.jsonl` per
//! SPEC §11.G.4. Supervision events flow:
//!
//! `supervisor` → `JsonlSupervisionPublisher::publish()` →
//!   - JSONL append to `data/supervision.jsonl` (rotating)
//!   - in-process broker fanout via `BusBroker::publish_local`
//!
//! Per SPEC §11.G.4: rotating 100 MB max, keep 10 archives
//! (`SUPERVISION_LOG_MAX_BYTES` × `SUPERVISION_LOG_ARCHIVE_COUNT`). Path
//! comes from `SUPERVISION_LOG_PATH = "data/supervision.jsonl"`.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::Serialize;
use titan_bus::BusBroker;
use titan_core::supervisor::event::{SupervisionEvent, SupervisionPublisher};

use titan_core::constants::{SUPERVISION_LOG_ARCHIVE_COUNT, SUPERVISION_LOG_MAX_BYTES};

/// Errors during supervision-log writing.
#[derive(Debug, thiserror::Error)]
pub enum SupervisionLogError {
    /// I/O failure.
    #[error("supervision_log I/O at {path}: {source}")]
    Io {
        /// Path attempted.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
}

/// Wire-format of one JSONL line (mirrors SPEC §11.G.4 example).
#[derive(Debug, Serialize)]
struct LogLine<'a> {
    ts: String,
    boot_generation: u64,
    event: &'a str,
    payload: &'a SupervisionEvent,
}

/// Rotating JSONL writer. Wrapped by `JsonlSupervisionPublisher`.
pub struct SupervisionLogWriter {
    path: PathBuf,
    boot_generation: u64,
    bytes_written: u64,
    file: Option<fs::File>,
}

impl SupervisionLogWriter {
    /// Open or create the supervision log at `path`.
    pub fn open(
        path: impl Into<PathBuf>,
        boot_generation: u64,
    ) -> Result<Self, SupervisionLogError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| SupervisionLogError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| SupervisionLogError::Io {
                path: path.clone(),
                source,
            })?;
        let bytes_written =
            file.metadata()
                .map(|m| m.len())
                .map_err(|source| SupervisionLogError::Io {
                    path: path.clone(),
                    source,
                })?;
        Ok(Self {
            path,
            boot_generation,
            bytes_written,
            file: Some(file),
        })
    }

    /// Append one event as a JSONL line. Rotates if appending would exceed
    /// `SUPERVISION_LOG_MAX_BYTES`.
    pub fn append(&mut self, event: &SupervisionEvent) -> Result<(), SupervisionLogError> {
        let line = LogLine {
            ts: chrono_iso8601_now(),
            boot_generation: self.boot_generation,
            event: event.msg_type(),
            payload: event,
        };
        let mut json = serde_json::to_vec(&line).map_err(|source| SupervisionLogError::Io {
            path: self.path.clone(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, source),
        })?;
        json.push(b'\n');

        if self.bytes_written + json.len() as u64 > SUPERVISION_LOG_MAX_BYTES {
            self.rotate()?;
        }

        let file = self.file.as_mut().ok_or_else(|| SupervisionLogError::Io {
            path: self.path.clone(),
            source: std::io::Error::other("writer closed"),
        })?;
        file.write_all(&json)
            .map_err(|source| SupervisionLogError::Io {
                path: self.path.clone(),
                source,
            })?;
        self.bytes_written += json.len() as u64;
        Ok(())
    }

    /// Rotate `.jsonl.<N>` chain: `.10 → DELETE`, `.9 → .10`, ..., `.1 → .2`,
    /// current `.jsonl → .1`, open fresh.
    fn rotate(&mut self) -> Result<(), SupervisionLogError> {
        // Close current file
        self.file = None;

        // Cascade: drop oldest, shift the rest
        let max = SUPERVISION_LOG_ARCHIVE_COUNT as usize;
        if max >= 1 {
            let oldest = self.path.with_extension(format!("jsonl.{max}"));
            if oldest.exists() {
                let _ = fs::remove_file(&oldest);
            }
            for i in (1..max).rev() {
                let from = self.path.with_extension(format!("jsonl.{i}"));
                let to = self.path.with_extension(format!("jsonl.{}", i + 1));
                if from.exists() {
                    fs::rename(&from, &to)
                        .map_err(|source| SupervisionLogError::Io { path: from, source })?;
                }
            }
            // current → .1
            let to = self.path.with_extension("jsonl.1");
            if self.path.exists() {
                fs::rename(&self.path, &to).map_err(|source| SupervisionLogError::Io {
                    path: self.path.clone(),
                    source,
                })?;
            }
        }

        // Reopen fresh file
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|source| SupervisionLogError::Io {
                path: self.path.clone(),
                source,
            })?;
        self.file = Some(file);
        self.bytes_written = 0;
        Ok(())
    }
}

fn chrono_iso8601_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now();
    let dur = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let nanos = dur.subsec_nanos();
    // Minimal RFC 3339 with nanosecond precision (no time zone offset; treat as UTC)
    unix_to_iso8601_utc(secs, nanos)
}

fn unix_to_iso8601_utc(unix_secs: u64, nanos: u32) -> String {
    // Stdlib doesn't have date math; use a minimal hand-rolled impl that
    // works for any year >= 1970.
    let days = (unix_secs / 86400) as i64;
    let secs_today = (unix_secs % 86400) as u32;
    let h = secs_today / 3600;
    let m = (secs_today % 3600) / 60;
    let s = secs_today % 60;

    // Civil-from-days algorithm (Howard Hinnant)
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m_civil = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m_civil <= 2 { y + 1 } else { y };

    format!(
        "{year:04}-{mc:02}-{d:02}T{h:02}:{m:02}:{s:02}.{ns:09}Z",
        year = year,
        mc = m_civil,
        d = d,
        h = h,
        m = m,
        s = s,
        ns = nanos,
    )
}

/// `SupervisionPublisher` impl that:
///   - appends to `data/supervision.jsonl` (rotating)
///   - publishes the event to the in-process broker
///
/// Errors writing the JSONL log are logged via `tracing` but do NOT fail
/// the publish — the bus message still goes out so live observers see it.
pub struct JsonlSupervisionPublisher {
    writer: Mutex<SupervisionLogWriter>,
    broker: Option<Arc<BusBroker>>,
    runtime: tokio::runtime::Handle,
}

impl JsonlSupervisionPublisher {
    /// New publisher writing to the given path + (optionally) bridging to
    /// a broker for live fanout.
    pub fn new(
        log_path: impl Into<PathBuf>,
        boot_generation: u64,
        broker: Option<Arc<BusBroker>>,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self, SupervisionLogError> {
        Ok(Self {
            writer: Mutex::new(SupervisionLogWriter::open(log_path, boot_generation)?),
            broker,
            runtime,
        })
    }
}

impl SupervisionPublisher for JsonlSupervisionPublisher {
    fn publish(&self, event: &SupervisionEvent) -> Result<(), String> {
        // 1. JSONL log (best-effort — supervision.jsonl is durable record;
        //    bus is observability)
        if let Err(e) = self.writer.lock().append(event) {
            tracing::warn!(
                err = ?e,
                event = event.msg_type(),
                "supervision_log: append failed"
            );
        }

        // 2. Bus fanout (best-effort — if broker isn't bound yet at very
        //    early boot, just log the event)
        if let Some(broker) = self.broker.clone() {
            let bytes = match build_event_payload(event) {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(err = %e, "supervision_log: msgpack encode failed");
                    return Ok(());
                }
            };
            let msg_type = event.msg_type().to_string();
            self.runtime.spawn(async move {
                broker
                    .publish_local(&msg_type, "kernel:supervisor", bytes)
                    .await;
            });
        }

        Ok(())
    }
}

fn build_event_payload(event: &SupervisionEvent) -> Result<Vec<u8>, String> {
    let json = serde_json::to_value(event).map_err(|e| format!("json: {e}"))?;
    let mp = serde_json_to_mpvalue(&json);
    let mut out = Vec::with_capacity(128);
    rmpv::encode::write_value(&mut out, &mp).map_err(|e| format!("rmpv: {e:?}"))?;
    Ok(out)
}

fn serde_json_to_mpvalue(v: &serde_json::Value) -> rmpv::Value {
    use rmpv::Value as Mp;
    match v {
        serde_json::Value::Null => Mp::Nil,
        serde_json::Value::Bool(b) => Mp::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Mp::Integer(i.into())
            } else if let Some(u) = n.as_u64() {
                Mp::Integer(u.into())
            } else if let Some(f) = n.as_f64() {
                Mp::F64(f)
            } else {
                Mp::Nil
            }
        }
        serde_json::Value::String(s) => Mp::String(s.as_str().into()),
        serde_json::Value::Array(arr) => Mp::Array(arr.iter().map(serde_json_to_mpvalue).collect()),
        serde_json::Value::Object(obj) => Mp::Map(
            obj.iter()
                .map(|(k, v)| (Mp::String(k.as_str().into()), serde_json_to_mpvalue(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;
    use titan_core::supervisor::types::SupervisionReason;

    fn fake_event() -> SupervisionEvent {
        SupervisionEvent::ChildDown {
            child_name: "test_child".into(),
            supervisor: "kernel".into(),
            reason: SupervisionReason::Panic,
            reason_detail: "test".into(),
            restart_count: 1,
            ts: SystemTime::now(),
        }
    }

    #[test]
    fn writer_appends_jsonl_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("supervision.jsonl");
        let mut w = SupervisionLogWriter::open(&path, 7).unwrap();
        w.append(&fake_event()).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("SUPERVISION_CHILD_DOWN"));
        assert!(content.contains("\"boot_generation\":7"));
        // Single line + trailing newline
        assert_eq!(content.matches('\n').count(), 1);
    }

    #[test]
    fn writer_rotates_at_size_threshold() {
        // Use a tempfile + tweak the threshold by writing many lines.
        // Since SUPERVISION_LOG_MAX_BYTES=100MB is too large to exercise
        // in a fast unit test, we verify the rotate() function directly.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("supervision.jsonl");
        let mut w = SupervisionLogWriter::open(&path, 1).unwrap();
        w.append(&fake_event()).unwrap();
        let initial_bytes = w.bytes_written;
        assert!(initial_bytes > 0);

        // Force rotate
        w.rotate().unwrap();

        // After rotate: .1 has the previous content, current is empty
        assert!(path.with_extension("jsonl.1").exists());
        assert_eq!(w.bytes_written, 0);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
    }

    #[test]
    fn iso8601_format_is_well_formed() {
        let s = chrono_iso8601_now();
        // Should match shape YYYY-MM-DDTHH:MM:SS.NNNNNNNNNZ
        assert!(s.len() >= 30, "{s} too short");
        assert!(s.ends_with('Z'));
        assert!(s.contains('T'));
    }

    #[test]
    fn jsonl_publisher_writes_on_publish() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("supervision.jsonl");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let pub_ = JsonlSupervisionPublisher::new(&path, 1, None, rt.handle().clone()).unwrap();
        pub_.publish(&fake_event()).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("SUPERVISION_CHILD_DOWN"));
    }
}
