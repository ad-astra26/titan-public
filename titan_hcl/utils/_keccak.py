"""Self-contained Keccak-256 (the pre-NIST padding used by Ethereum / Light
Protocol address derivation — NOT `hashlib.sha3_256`, which uses NIST 0x06
padding and produces different digests).

Pure-Python so it deploys fleet-wide via a plain `git pull` with NO new pip
dependency (a missing dep on T2/T3 would silently break the ZK-Vault genesis
create). Validated byte-exact against the light-sdk-types v1 `derive_address`
test vectors (`tests/test_zk_vault_sovereign_state.py`).

Used by `solana_client.derive_light_v1_address` to compute a Titan's canonical
SovereignState compressed-account address client-side (needed to request the
genesis non-inclusion proof from Photon). Light v1 `derive_address` uses Keccak
(`light-sdk-types-0.23/src/address.rs:38-55`), not Poseidon.
"""
from __future__ import annotations

_RC = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)
_ROT = (
    (0, 36, 3, 41, 18), (1, 44, 10, 45, 2), (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56), (27, 20, 39, 8, 14),
)
_MASK = (1 << 64) - 1


def _rotl(x: int, n: int) -> int:
    return ((x << n) | (x >> (64 - n))) & _MASK


def _keccak_f(a: list) -> None:
    for rc in _RC:
        # θ
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rotl(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                a[x][y] ^= d[x]
        # ρ and π
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rotl(a[x][y], _ROT[x][y])
        # χ
        for x in range(5):
            for y in range(5):
                a[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y])
        # ι
        a[0][0] ^= rc


def keccak256(data: bytes) -> bytes:
    """Keccak-256 (rate 1088 bits / 136 bytes, capacity 512, pad 0x01…0x80)."""
    rate = 136
    a = [[0] * 5 for _ in range(5)]
    # absorb
    msg = bytearray(data)
    msg.append(0x01)
    while len(msg) % rate != 0:
        msg.append(0x00)
    msg[-1] ^= 0x80
    for off in range(0, len(msg), rate):
        block = msg[off:off + rate]
        for i in range(rate // 8):
            lane = int.from_bytes(block[i * 8:i * 8 + 8], "little")
            a[i % 5][i // 5] ^= lane
        _keccak_f(a)
    # squeeze (256 bits ≤ rate → single block)
    out = bytearray()
    for i in range(4):  # 4 lanes × 8 bytes = 32 bytes
        out += (a[i % 5][i // 5]).to_bytes(8, "little")
    return bytes(out[:32])
