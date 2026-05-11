//! payload — Typed encoder/decoder for the 256-byte fastbus slot convention.
//!
//! Per PLAN §9.4 + SPEC §9.A trinity-rs ↔ kernel fastbus protocol:
//!
//! - kernel → substrate: `Circadian`, `PiHeartbeat` (clock validation + Schumann phase modulation)
//! - substrate → kernel: `SchumannEpoch` (Schumann tick visibility for kernel-side clock validation)
//!
//! # Slot layout (per PLAN §9.4)
//!
//! 256 bytes total per slot:
//!
//! ```text
//! [0:1]    uint8   msg_type        — `MsgType` enum discriminant (1..255)
//! [1:9]    uint64  ts_ns           — wall-clock nanoseconds (little-endian)
//! [9:17]   uint64  epoch           — producer's monotonic epoch counter
//! [17:25]  uint64  producer_pid    — producer process PID (cross-process traceability)
//! [25:256] uint8[231] payload      — message-type-specific (zero-padded)
//! ```
//!
//! All multi-byte fields are little-endian (matches SPEC §7.4 endianness lock).
//!
//! # Why 256 bytes
//!
//! Matches L1 cache line stride on x86_64 (typically 64 bytes; 256B = 4 lines).
//! One slot = one cache transaction round-trip on the consumer side.

use thiserror::Error;

use crate::FASTBUS_SLOT_BYTES;

const SLOT_BYTES: usize = FASTBUS_SLOT_BYTES as usize;

/// Header byte offset constants — used by typed encoders + parity vectors.
pub const OFFSET_MSG_TYPE: usize = 0;
/// `ts_ns` field offset.
pub const OFFSET_TS_NS: usize = 1;
/// `epoch` field offset.
pub const OFFSET_EPOCH: usize = 9;
/// `producer_pid` field offset.
pub const OFFSET_PRODUCER_PID: usize = 17;
/// Payload start offset (after 25-byte header).
pub const PAYLOAD_OFFSET: usize = 25;
/// Payload byte length (`SLOT_BYTES - PAYLOAD_OFFSET`).
pub const PAYLOAD_LEN: usize = SLOT_BYTES - PAYLOAD_OFFSET;

const _: () = assert!(SLOT_BYTES == 256);
const _: () = assert!(PAYLOAD_OFFSET == 25);
const _: () = assert!(PAYLOAD_LEN == 231);

/// Message type discriminant per PLAN §9.4. Reserved 0 = invalid;
/// 1–255 = active type space (room for future Phase D additions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MsgType {
    /// Kernel → substrate: 1 Hz circadian tick. Substrate uses for Schumann
    /// phase modulation (drift correction relative to kernel's wall clock).
    Circadian = 1,
    /// Kernel → substrate: ~3 Hz π-heartbeat tick. Substrate uses for
    /// consciousness-epoch tracking + as input to local timer wheel.
    PiHeartbeat = 2,
    /// Substrate → kernel: Schumann epoch boundary (every 9 spirit ticks
    /// = one body cycle). Kernel uses for cross-checking its own clocks
    /// against the substrate-driven Schumann wheel.
    SchumannEpoch = 3,
}

impl MsgType {
    /// Discriminant byte.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Decode from byte; `None` if not a registered variant.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(MsgType::Circadian),
            2 => Some(MsgType::PiHeartbeat),
            3 => Some(MsgType::SchumannEpoch),
            _ => None,
        }
    }

    /// Canonical name (logging + diagnostics).
    pub fn as_str(self) -> &'static str {
        match self {
            MsgType::Circadian => "circadian",
            MsgType::PiHeartbeat => "pi_heartbeat",
            MsgType::SchumannEpoch => "schumann_epoch",
        }
    }
}

/// Decoded fastbus message — typed view over the 256-byte slot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Message {
    /// Message type per the kernel ↔ substrate protocol.
    pub msg_type: MsgType,
    /// Wall-clock nanoseconds at producer side.
    pub ts_ns: u64,
    /// Producer's monotonic epoch counter (semantics depend on `msg_type`).
    pub epoch: u64,
    /// Producer process PID — useful for cross-process traceability.
    pub producer_pid: u64,
    /// Payload bytes (231 bytes; zero-padded if message-type-specific
    /// payload is shorter). Most C-S3 messages don't use the payload.
    pub payload: [u8; PAYLOAD_LEN],
}

/// Encode/decode errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PayloadError {
    /// Reserved (zero) or unrecognized message-type byte.
    #[error("fastbus: unknown msg_type byte = {got}")]
    UnknownMsgType {
        /// Observed byte.
        got: u8,
    },
}

impl Message {
    /// Construct a header-only message (zero payload).
    pub fn new(msg_type: MsgType, ts_ns: u64, epoch: u64, producer_pid: u64) -> Self {
        Self {
            msg_type,
            ts_ns,
            epoch,
            producer_pid,
            payload: [0u8; PAYLOAD_LEN],
        }
    }

    /// Encode to a 256-byte slot ready for [`crate::Producer::publish`].
    pub fn encode(&self) -> [u8; SLOT_BYTES] {
        let mut out = [0u8; SLOT_BYTES];
        out[OFFSET_MSG_TYPE] = self.msg_type.as_u8();
        out[OFFSET_TS_NS..OFFSET_TS_NS + 8].copy_from_slice(&self.ts_ns.to_le_bytes());
        out[OFFSET_EPOCH..OFFSET_EPOCH + 8].copy_from_slice(&self.epoch.to_le_bytes());
        out[OFFSET_PRODUCER_PID..OFFSET_PRODUCER_PID + 8]
            .copy_from_slice(&self.producer_pid.to_le_bytes());
        out[PAYLOAD_OFFSET..].copy_from_slice(&self.payload);
        out
    }

    /// Decode from a 256-byte slot received via [`crate::Consumer::recv_and_commit`].
    pub fn decode(bytes: &[u8; SLOT_BYTES]) -> Result<Self, PayloadError> {
        let msg_type =
            MsgType::from_u8(bytes[OFFSET_MSG_TYPE]).ok_or(PayloadError::UnknownMsgType {
                got: bytes[OFFSET_MSG_TYPE],
            })?;
        let ts_ns = u64::from_le_bytes(bytes[OFFSET_TS_NS..OFFSET_TS_NS + 8].try_into().unwrap());
        let epoch = u64::from_le_bytes(bytes[OFFSET_EPOCH..OFFSET_EPOCH + 8].try_into().unwrap());
        let producer_pid = u64::from_le_bytes(
            bytes[OFFSET_PRODUCER_PID..OFFSET_PRODUCER_PID + 8]
                .try_into()
                .unwrap(),
        );
        let mut payload = [0u8; PAYLOAD_LEN];
        payload.copy_from_slice(&bytes[PAYLOAD_OFFSET..]);
        Ok(Self {
            msg_type,
            ts_ns,
            epoch,
            producer_pid,
            payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msg_type_round_trip_all_variants() {
        for &mt in &[
            MsgType::Circadian,
            MsgType::PiHeartbeat,
            MsgType::SchumannEpoch,
        ] {
            let byte = mt.as_u8();
            assert_eq!(MsgType::from_u8(byte), Some(mt));
        }
    }

    #[test]
    fn msg_type_zero_byte_is_invalid() {
        assert_eq!(MsgType::from_u8(0), None);
    }

    #[test]
    fn msg_type_unknown_byte_returns_none() {
        for byte in 4..=255u8 {
            assert_eq!(MsgType::from_u8(byte), None, "byte {byte}");
        }
    }

    #[test]
    fn msg_type_canonical_strings() {
        assert_eq!(MsgType::Circadian.as_str(), "circadian");
        assert_eq!(MsgType::PiHeartbeat.as_str(), "pi_heartbeat");
        assert_eq!(MsgType::SchumannEpoch.as_str(), "schumann_epoch");
    }

    #[test]
    fn encode_decode_round_trip() {
        let m = Message::new(MsgType::Circadian, 1_700_000_000_000_000_000, 42, 12345);
        let bytes = m.encode();
        let decoded = Message::decode(&bytes).unwrap();
        assert_eq!(m, decoded);
    }

    #[test]
    fn encode_layout_matches_plan_section_94() {
        // PLAN §9.4: msg_type[0:1] + ts_ns[1:9] + epoch[9:17] + producer_pid[17:25]
        let m = Message::new(
            MsgType::PiHeartbeat,
            0xDEAD_BEEF_CAFE_BABE,
            0x0123_4567,
            0xABCD,
        );
        let bytes = m.encode();
        // msg_type byte
        assert_eq!(bytes[0], 2u8); // MsgType::PiHeartbeat = 2
                                   // ts_ns LE
        assert_eq!(
            u64::from_le_bytes(bytes[1..9].try_into().unwrap()),
            0xDEAD_BEEF_CAFE_BABE
        );
        // epoch LE
        assert_eq!(
            u64::from_le_bytes(bytes[9..17].try_into().unwrap()),
            0x0123_4567
        );
        // producer_pid LE
        assert_eq!(
            u64::from_le_bytes(bytes[17..25].try_into().unwrap()),
            0xABCD
        );
        // Payload zero-padded
        for v in &bytes[25..] {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn decode_unknown_msg_type_returns_error() {
        let mut bytes = [0u8; SLOT_BYTES];
        bytes[0] = 99; // not a valid variant
        let err = Message::decode(&bytes).unwrap_err();
        assert_eq!(err, PayloadError::UnknownMsgType { got: 99 });
    }

    #[test]
    fn decode_zero_byte_returns_error() {
        // All-zero slot (initial state) → msg_type=0 = invalid
        let bytes = [0u8; SLOT_BYTES];
        let err = Message::decode(&bytes).unwrap_err();
        assert_eq!(err, PayloadError::UnknownMsgType { got: 0 });
    }

    #[test]
    fn payload_offset_25_payload_len_231() {
        assert_eq!(PAYLOAD_OFFSET, 25);
        assert_eq!(PAYLOAD_LEN, 231);
        assert_eq!(PAYLOAD_OFFSET + PAYLOAD_LEN, 256);
    }

    #[test]
    fn encode_preserves_payload_bytes() {
        let mut m = Message::new(MsgType::SchumannEpoch, 100, 7, 999);
        m.payload[0] = 0x42;
        m.payload[100] = 0x99;
        m.payload[230] = 0xFF;
        let bytes = m.encode();
        let decoded = Message::decode(&bytes).unwrap();
        assert_eq!(decoded.payload[0], 0x42);
        assert_eq!(decoded.payload[100], 0x99);
        assert_eq!(decoded.payload[230], 0xFF);
    }
}
