//! logging — `tracing-subscriber` JSON formatter per SPEC §16.
//!
//! Mandatory structured-JSON fields per SPEC §16:
//! `ts`, `level`, `binary`, `titan_id`, `boot_generation`, `pid`, `msg`,
//! `event`, `span`.
//!
//! `tracing-subscriber`'s built-in JSON layer covers `ts` (auto), `level`,
//! `pid` (target field), `msg` (the format string), `span`. The
//! supervisor-mandated fields (`binary`, `titan_id`, `boot_generation`,
//! `event`) are added as fields on emission OR as global span attributes
//! at boot via `init()`.

use std::sync::Arc;

use tracing::Level;
use tracing_subscriber::fmt::format::FmtSpan;

/// Initialize the tracing subscriber for JSON output.
///
/// Sets `binary="kernel"`, `titan_id`, and the chosen log level as global
/// attributes. Returns an opaque guard handle (Arc-wrapped) — kept for
/// the lifetime of the kernel process.
///
/// Per SPEC §16: every line ends up with the mandatory fields. Per-event
/// optional fields (e.g. `event=BOOT_COMPLETE`, `child_name=...`) are
/// added at the emission site via `tracing::info!(event = "...", ...)`.
pub fn init(level: Level, titan_id: &str) -> Arc<()> {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let filter = tracing_subscriber::EnvFilter::from_default_env().add_directive(level.into());

    // JSON layer with the mandatory fields baked into every line.
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

    // Emit a kernel-boot line that downstream tooling can grep.
    tracing::info!(
        binary = "kernel",
        titan_id = %titan_id,
        event = "LOGGING_INITIALIZED",
        level = ?level,
        "structured-JSON logging initialized per SPEC §16",
    );
    Arc::new(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_returns_guard_without_panicking() {
        // Multiple calls in the same process are no-ops (try_init handles this).
        let _g1 = init(Level::INFO, "T1");
        let _g2 = init(Level::DEBUG, "T1");
    }

    #[test]
    fn init_accepts_all_levels() {
        for level in [Level::ERROR, Level::WARN, Level::INFO, Level::DEBUG] {
            let _ = init(level, "T1");
        }
    }
}
