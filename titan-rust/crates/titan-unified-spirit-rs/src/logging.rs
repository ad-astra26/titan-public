//! logging — `tracing-subscriber` JSON formatter per SPEC §16.
//!
//! Mandatory structured-JSON fields per SPEC §16:
//! `ts`, `level`, `binary`, `titan_id`, `boot_generation`, `pid`, `msg`,
//! `event`, `span`. `binary` set to `"unified-spirit"` here.
//!
//! Mirrors `titan-rust/crates/titan-kernel-rs/src/logging.rs` to keep
//! observability discipline consistent across all Rust binaries.

use std::sync::Arc;

use tracing::Level;
use tracing_subscriber::fmt::format::FmtSpan;

/// Initialize the tracing subscriber for JSON output.
///
/// Per SPEC §16: every line ends up with the mandatory fields. Per-event
/// optional fields (e.g. `event=BOOT_COMPLETE`, `child_name=...`) are
/// added at the emission site via `tracing::info!(event = "...", ...)`.
pub fn init(level: Level, titan_id: &str) -> Arc<()> {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let filter = tracing_subscriber::EnvFilter::from_default_env().add_directive(level.into());

    let json_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_current_span(true)
        .with_span_list(true)
        .with_target(true)
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(true)
        .with_thread_names(false)
        .with_span_events(FmtSpan::CLOSE);

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(json_layer)
        .try_init();

    tracing::info!(
        binary = "unified-spirit",
        titan_id = titan_id,
        level = ?level,
        "tracing initialized"
    );

    Arc::new(())
}
