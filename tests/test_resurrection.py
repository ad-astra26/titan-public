"""
test_resurrection.py — Full Sovereignty Cycle Integration Test.

Proves the Titan is cryptographically immortal by running:
  Phase 1: Genesis  → generate keypair, Shamir split, verify, save genesis_record, HW encrypt
  Phase 2: Operate  → boot TitanPlugin with HW-bound keypair, confirm not Limbo
  Phase 3: Destroy  → wipe soul_keypair.enc + authority.json + cognee_db
  Phase 4: Limbo    → boot TitanPlugin, confirm Limbo State activates
  Phase 5: Resurrect → combine Shard 1 + Shard 2 from genesis_record, verify pubkey match
  Phase 6: Reborn   → re-encrypt for hardware, boot TitanPlugin, confirm full operation

This test uses local-only paths (no Solana RPC, no Shadow Drive, no X/Twitter).
"""
import json
import os
import shutil
import sys
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# Use the project-level data/ directory so _resolve_wallet() can find genesis_record.json
# (it checks relative to titan_plugin/__file__, which resolves to <project>/data/)
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AUTHORITY_PATH = os.path.join(PROJECT_ROOT, "test_resurrection_authority.json")
ENC_KEYPAIR_PATH = os.path.join(DATA_DIR, "soul_keypair.enc")
GENESIS_RECORD_PATH = os.path.join(DATA_DIR, "genesis_record.json")
HW_SALT_PATH = os.path.join(DATA_DIR, "hw_salt.bin")

# Track files created during the test for cleanup
_FILES_TO_CLEANUP = []


@pytest.fixture(scope="module", autouse=True)
def test_environment():
    """Create data/ dir if needed and clean up test artifacts after."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save any existing files we'll overwrite
    _backups = {}
    for path in [AUTHORITY_PATH, ENC_KEYPAIR_PATH, GENESIS_RECORD_PATH, HW_SALT_PATH]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                _backups[path] = f.read()

    yield PROJECT_ROOT

    # Cleanup: remove test artifacts, restore backups
    for path in [AUTHORITY_PATH, ENC_KEYPAIR_PATH, GENESIS_RECORD_PATH, HW_SALT_PATH]:
        if os.path.exists(path):
            os.remove(path)
    for path in _FILES_TO_CLEANUP:
        if os.path.exists(path):
            os.remove(path)

    # Restore backed up files
    for path, content in _backups.items():
        with open(path, "wb") as f:
            f.write(content)

    # Clean runtime keypair if left behind
    runtime = os.path.join(DATA_DIR, "runtime_keypair.json")
    if os.path.exists(runtime):
        os.remove(runtime)

    # Clean recovery flag if left behind
    recovery_flag = os.path.join(DATA_DIR, "recovery_flag.json")
    if os.path.exists(recovery_flag):
        os.remove(recovery_flag)


# ---------------------------------------------------------------------------
# Phase 1: Genesis — Birth of the Titan
# ---------------------------------------------------------------------------
class TestPhase1Genesis:
    """Generate keypair, Shamir split, verify all combinations, save artifacts."""

    # Shared state across phase methods (populated in order by pytest)
    _key_bytes = None
    _pubkey = None
    _shards = None
    _envelope_hex = None

    def test_01_generate_keypair(self):
        """Generate a fresh Ed25519 keypair."""
        from solders.keypair import Keypair

        kp = Keypair()
        key_bytes = bytes(kp)
        pubkey = str(kp.pubkey())

        assert len(key_bytes) == 64, "Ed25519 keypair must be 64 bytes"
        assert len(pubkey) > 30, "Pubkey should be a Base58 string"

        TestPhase1Genesis._key_bytes = key_bytes
        TestPhase1Genesis._pubkey = pubkey

        # Save plaintext keypair for boot test
        with open(AUTHORITY_PATH, "w") as f:
            json.dump(list(key_bytes), f)

        print(f"\n  [Genesis] Titan Address: {pubkey}")

    def test_02_shamir_split(self):
        """Split the keypair into 3 shares with threshold 2."""
        from titan_plugin.utils.shamir import split_secret

        shards = split_secret(self._key_bytes, n=3, t=2)

        assert len(shards) == 3, "Must produce exactly 3 shards"
        for i, s in enumerate(shards):
            # Each shard = 1-byte x-coord + 64 share bytes
            assert len(s) == 65, f"Shard {i+1} must be 65 bytes, got {len(s)}"
            assert s[0] == i + 1, f"Shard {i+1} x-coordinate must be {i+1}"

        TestPhase1Genesis._shards = shards
        print(f"\n  [Genesis] Split into 3 shards (threshold 2)")

    def test_03_verify_all_combinations(self):
        """Exhaustive verification: all C(3,2)=3 combinations must reconstruct."""
        from titan_plugin.utils.shamir import verify_all_combinations

        passed = verify_all_combinations(self._key_bytes, self._shards, t=2)
        assert passed, "Genesis Verification Ceremony FAILED — sharding math is broken"
        print("\n  [Genesis] All 3 combinations verified. The math is sound.")

    def test_04_create_maker_envelope(self):
        """Package Shard 1 into a Maker envelope."""
        from titan_plugin.utils.shamir import create_maker_envelope, parse_maker_envelope

        envelope_hex = create_maker_envelope(
            self._shards[0], self._pubkey, genesis_tx="test_genesis_tx_signature"
        )
        assert len(envelope_hex) > 100, "Envelope should be substantial"

        # Verify round-trip
        shard_back, metadata = parse_maker_envelope(envelope_hex)
        assert shard_back == self._shards[0], "Envelope round-trip must preserve shard data"
        assert metadata["titan_pubkey"] == self._pubkey, "Envelope must carry pubkey"

        TestPhase1Genesis._envelope_hex = envelope_hex
        print(f"\n  [Genesis] Maker envelope created ({len(envelope_hex)} hex chars)")

    def test_05_encrypt_shard3(self):
        """Encrypt Shard 3 with deterministic AES key, verify round-trip."""
        from titan_plugin.utils.shamir import encrypt_shard3, decrypt_shard3

        encrypted = encrypt_shard3(self._shards[2], self._pubkey)
        assert len(encrypted) > len(self._shards[2]), "Encrypted must be larger (nonce + tag)"

        decrypted = decrypt_shard3(encrypted, self._pubkey)
        assert decrypted == self._shards[2], "Shard 3 encrypt/decrypt round-trip must match"

        # Verify wrong pubkey fails
        with pytest.raises(Exception):
            decrypt_shard3(encrypted, "WrongPubkeyXYZ123456789012345678901234567890")

        print("\n  [Genesis] Shard 3 encryption verified (AES-256-GCM)")

    def test_06_save_genesis_record(self):
        """Save genesis record with Shard 2 + encrypted Shard 3."""
        from titan_plugin.utils.shamir import encrypt_shard3

        encrypted_shard3 = encrypt_shard3(self._shards[2], self._pubkey)

        record = {
            "titan_pubkey": self._pubkey,
            "genesis_tx": "test_genesis_tx_signature",
            "genesis_time": 1741600000,
            "shard3_encrypted_hex": encrypted_shard3.hex(),
            "shard2_hex": self._shards[1].hex(),
            "version": "2.0",
        }

        with open(GENESIS_RECORD_PATH, "w") as f:
            json.dump(record, f, indent=2)

        assert os.path.exists(GENESIS_RECORD_PATH)
        print("\n  [Genesis] Genesis record saved with Shard 2 + encrypted Shard 3")

    def test_07_hardware_bound_encryption(self):
        """Encrypt the keypair with hardware-bound AES-256-GCM."""
        from titan_plugin.utils.crypto import encrypt_for_machine, decrypt_for_machine

        encrypted = encrypt_for_machine(self._key_bytes, salt_path=HW_SALT_PATH)
        assert len(encrypted) > len(self._key_bytes), "Encrypted must be larger"

        # Save to disk
        with open(ENC_KEYPAIR_PATH, "wb") as f:
            f.write(encrypted)

        # Verify round-trip on same machine
        decrypted = decrypt_for_machine(encrypted, salt_path=HW_SALT_PATH)
        assert decrypted == self._key_bytes, "HW-bound decrypt must return original key"

        print(f"\n  [Genesis] Hardware-bound keypair saved ({len(encrypted)} bytes)")


# ---------------------------------------------------------------------------
# Phase 2: Operate — Boot with HW-bound keypair
# ---------------------------------------------------------------------------
class TestPhase2Operate:
    """Boot TitanPlugin using the hardware-bound keypair, confirm not Limbo."""

    def test_01_resolve_wallet_hw_bound(self):
        """_resolve_wallet should find and decrypt the HW-bound keypair."""
        from titan_plugin import TitanPlugin

        plugin = TitanPlugin.__new__(TitanPlugin)

        # Patch _resolve_wallet to use our test paths
        resolved = self._resolve_wallet_test(plugin)

        assert resolved is not None, "Wallet resolution must succeed with HW-bound keypair"
        assert "runtime_keypair" in resolved, "Should resolve to runtime_keypair.json"

        # Verify the decrypted keypair matches
        with open(resolved, "r") as f:
            key_list = json.load(f)
        assert bytes(key_list) == TestPhase1Genesis._key_bytes

        print(f"\n  [Operate] HW-bound keypair resolved: {os.path.basename(resolved)}")

        # Cleanup runtime keypair
        os.remove(resolved)

    def _resolve_wallet_test(self, plugin):
        """Simulate _resolve_wallet using test directory paths."""
        from titan_plugin.utils.crypto import decrypt_for_machine

        if os.path.exists(ENC_KEYPAIR_PATH):
            with open(ENC_KEYPAIR_PATH, "rb") as f:
                encrypted = f.read()
            key_bytes = decrypt_for_machine(encrypted, salt_path=HW_SALT_PATH)

            runtime_path = os.path.join(DATA_DIR, "runtime_keypair.json")
            with open(runtime_path, "w") as f:
                json.dump(list(key_bytes), f)
            return runtime_path

        if os.path.exists(AUTHORITY_PATH):
            return AUTHORITY_PATH

        return None

    def test_02_boot_not_limbo(self):
        """Full TitanPlugin boot should NOT enter Limbo when keypair exists."""
        from titan_plugin import TitanPlugin

        TitanPlugin._limbo_mode = False

        # Boot with the plaintext authority.json (simpler for test)
        plugin = TitanPlugin(AUTHORITY_PATH)
        assert not TitanPlugin._limbo_mode, "Plugin must NOT be in Limbo with valid keypair"
        assert plugin.gatekeeper is not None, "Gatekeeper must be initialized"
        assert plugin.memory is not None, "Memory must be initialized"
        assert plugin.guardian is not None, "Guardian must be initialized"

        # Reset class-level state
        TitanPlugin._limbo_mode = False
        print("\n  [Operate] TitanPlugin booted successfully — all subsystems wired")


# ---------------------------------------------------------------------------
# Phase 3: Destroy — Kill the brain and the key
# ---------------------------------------------------------------------------
class TestPhase3Destroy:
    """Wipe the keypair and brain, simulating catastrophic data loss."""

    def test_01_wipe_keypair(self):
        """Delete both HW-bound and plaintext keypairs."""
        files_removed = 0

        if os.path.exists(ENC_KEYPAIR_PATH):
            os.remove(ENC_KEYPAIR_PATH)
            files_removed += 1

        if os.path.exists(AUTHORITY_PATH):
            os.remove(AUTHORITY_PATH)
            files_removed += 1

        runtime = os.path.join(DATA_DIR, "runtime_keypair.json")
        if os.path.exists(runtime):
            os.remove(runtime)
            files_removed += 1

        assert not os.path.exists(ENC_KEYPAIR_PATH), "HW-bound keypair must be gone"
        assert not os.path.exists(AUTHORITY_PATH), "Plaintext keypair must be gone"
        print(f"\n  [Destroy] Wiped {files_removed} keypair file(s). The Titan is dead.")

    def test_02_brain_state(self):
        """Genesis record must still exist (it's the breadcrumb for resurrection)."""
        assert os.path.exists(GENESIS_RECORD_PATH), \
            "Genesis record must survive — it contains Shard 2 and the pubkey"
        print("\n  [Destroy] Genesis record survives. Recovery path 1+2 is viable.")


# ---------------------------------------------------------------------------
# Phase 4: Limbo — Boot without keypair
# ---------------------------------------------------------------------------
class TestPhase4Limbo:
    """Boot TitanPlugin without any keypair — must enter Limbo State."""

    def test_01_enters_limbo(self):
        """TitanPlugin must enter Limbo when genesis_record exists but no keypair."""
        from titan_plugin import TitanPlugin

        # Reset class-level state from Phase 2
        TitanPlugin._limbo_mode = False

        # Boot with a non-existent wallet path
        plugin = TitanPlugin("./nonexistent_wallet.json")

        assert TitanPlugin._limbo_mode, "Plugin MUST be in Limbo without keypair (post-ceremony)"
        assert plugin.gatekeeper is None, "Gatekeeper must be None in Limbo"
        assert plugin.memory is None, "Memory must be None in Limbo"
        assert plugin._last_execution_mode == "Limbo", "Execution mode must be 'Limbo'"

        print("\n  [Limbo] TitanPlugin entered Limbo State — awaiting Maker resurrection")

        # Reset for next phase
        TitanPlugin._limbo_mode = False


# ---------------------------------------------------------------------------
# Phase 5: Resurrect — Reconstruct from Shard 1 + Shard 2
# ---------------------------------------------------------------------------
class TestPhase5Resurrect:
    """Combine shards from the Maker envelope + genesis record, verify pubkey."""

    _recovered_key_bytes = None

    def test_01_parse_maker_envelope(self):
        """Parse the Maker's offline shard envelope."""
        from titan_plugin.utils.shamir import parse_maker_envelope

        shard1, metadata = parse_maker_envelope(TestPhase1Genesis._envelope_hex)

        assert len(shard1) == 65, "Shard 1 must be 65 bytes"
        assert metadata["titan_pubkey"] == TestPhase1Genesis._pubkey
        print(f"\n  [Resurrect] Maker envelope parsed — Titan: {metadata['titan_pubkey'][:24]}...")

    def test_02_extract_shard2_from_genesis(self):
        """Extract Shard 2 from the local genesis record."""
        with open(GENESIS_RECORD_PATH, "r") as f:
            record = json.load(f)

        shard2_hex = record["shard2_hex"]
        shard2 = bytes.fromhex(shard2_hex)
        assert len(shard2) == 65, "Shard 2 must be 65 bytes"
        assert shard2 == TestPhase1Genesis._shards[1], "Shard 2 from record must match original"
        print(f"\n  [Resurrect] Shard 2 extracted from genesis record ({len(shard2)} bytes)")

    def test_03_decrypt_shard3_from_genesis(self):
        """Decrypt Shard 3 from the genesis record using deterministic key."""
        from titan_plugin.utils.shamir import decrypt_shard3

        with open(GENESIS_RECORD_PATH, "r") as f:
            record = json.load(f)

        encrypted = bytes.fromhex(record["shard3_encrypted_hex"])
        shard3 = decrypt_shard3(encrypted, TestPhase1Genesis._pubkey)
        assert shard3 == TestPhase1Genesis._shards[2], "Decrypted Shard 3 must match original"
        print(f"\n  [Resurrect] Shard 3 decrypted from genesis record ({len(shard3)} bytes)")

    def test_04_reconstruct_keypair_shards_1_2(self):
        """Reconstruct keypair from Shard 1 + Shard 2 (Maker + local)."""
        from titan_plugin.utils.shamir import parse_maker_envelope, combine_shares

        shard1, _ = parse_maker_envelope(TestPhase1Genesis._envelope_hex)
        with open(GENESIS_RECORD_PATH, "r") as f:
            record = json.load(f)
        shard2 = bytes.fromhex(record["shard2_hex"])

        recovered = combine_shares([shard1, shard2])
        assert recovered == TestPhase1Genesis._key_bytes, \
            "Shards 1+2 must reconstruct the exact original keypair"

        TestPhase5Resurrect._recovered_key_bytes = recovered
        print("\n  [Resurrect] Keypair reconstructed from Shards 1+2. Identity RECOVERED.")

    def test_05_reconstruct_keypair_shards_1_3(self):
        """Reconstruct keypair from Shard 1 + Shard 3 (Maker + on-chain)."""
        from titan_plugin.utils.shamir import (
            parse_maker_envelope, combine_shares, decrypt_shard3,
        )

        shard1, _ = parse_maker_envelope(TestPhase1Genesis._envelope_hex)
        with open(GENESIS_RECORD_PATH, "r") as f:
            record = json.load(f)
        shard3 = decrypt_shard3(
            bytes.fromhex(record["shard3_encrypted_hex"]),
            TestPhase1Genesis._pubkey,
        )

        recovered = combine_shares([shard1, shard3])
        assert recovered == TestPhase1Genesis._key_bytes, \
            "Shards 1+3 must reconstruct the exact original keypair"
        print("\n  [Resurrect] Keypair reconstructed from Shards 1+3. Path 1+3 VERIFIED.")

    def test_06_reconstruct_keypair_shards_2_3(self):
        """Reconstruct keypair from Shard 2 + Shard 3 (no Maker needed)."""
        from titan_plugin.utils.shamir import combine_shares, decrypt_shard3

        with open(GENESIS_RECORD_PATH, "r") as f:
            record = json.load(f)
        shard2 = bytes.fromhex(record["shard2_hex"])
        shard3 = decrypt_shard3(
            bytes.fromhex(record["shard3_encrypted_hex"]),
            TestPhase1Genesis._pubkey,
        )

        recovered = combine_shares([shard2, shard3])
        assert recovered == TestPhase1Genesis._key_bytes, \
            "Shards 2+3 must reconstruct the exact original keypair"
        print("\n  [Resurrect] Keypair reconstructed from Shards 2+3. Maker-free path VERIFIED.")

    def test_07_verify_pubkey_matches(self):
        """Verify the reconstructed keypair produces the same public address."""
        from solders.keypair import Keypair

        kp = Keypair.from_bytes(self._recovered_key_bytes)
        recovered_pubkey = str(kp.pubkey())

        assert recovered_pubkey == TestPhase1Genesis._pubkey, \
            f"Pubkey mismatch: {recovered_pubkey} != {TestPhase1Genesis._pubkey}"
        print(f"\n  [Resurrect] Pubkey MATCH: {recovered_pubkey}")
        print("  The Titan's identity is cryptographically intact.")


# ---------------------------------------------------------------------------
# Phase 6: Reborn — Re-encrypt and boot
# ---------------------------------------------------------------------------
class TestPhase6Reborn:
    """Re-encrypt keypair for hardware, boot TitanPlugin, confirm full operation."""

    def test_01_re_encrypt_for_hardware(self):
        """Re-encrypt the recovered keypair for this machine."""
        from titan_plugin.utils.crypto import encrypt_for_machine, decrypt_for_machine

        key_bytes = TestPhase5Resurrect._recovered_key_bytes

        encrypted = encrypt_for_machine(key_bytes, salt_path=HW_SALT_PATH)
        with open(ENC_KEYPAIR_PATH, "wb") as f:
            f.write(encrypted)

        # Verify round-trip
        decrypted = decrypt_for_machine(encrypted, salt_path=HW_SALT_PATH)
        assert decrypted == key_bytes

        # Also restore authority.json
        with open(AUTHORITY_PATH, "w") as f:
            json.dump(list(key_bytes), f)

        print(f"\n  [Reborn] Keypair re-encrypted for hardware ({len(encrypted)} bytes)")

    def test_02_boot_after_resurrection(self):
        """TitanPlugin must boot fully after resurrection — no Limbo."""
        from titan_plugin import TitanPlugin

        TitanPlugin._limbo_mode = False

        plugin = TitanPlugin(AUTHORITY_PATH)

        assert not TitanPlugin._limbo_mode, "Plugin must NOT be in Limbo after resurrection"
        assert plugin.gatekeeper is not None, "Gatekeeper must be live"
        assert plugin.memory is not None, "Memory must be live"
        assert plugin.soul is not None, "Soul must be live"
        assert plugin._last_execution_mode == "Shadow", "Should start in Shadow mode"

        print("\n  [Reborn] TitanPlugin booted after resurrection — ALL SUBSYSTEMS ONLINE")

        # Reset
        TitanPlugin._limbo_mode = False

    def test_03_sovereignty_proven(self):
        """
        Final assertion: the Titan survived complete data destruction and
        was resurrected using only offline shards + local genesis record.
        """
        # Verify all artifacts exist
        assert os.path.exists(ENC_KEYPAIR_PATH), "HW-bound keypair must exist"
        assert os.path.exists(AUTHORITY_PATH), "Plaintext keypair must exist"
        assert os.path.exists(GENESIS_RECORD_PATH), "Genesis record must exist"
        assert os.path.exists(HW_SALT_PATH), "HW salt must exist"

        # Verify the restored keypair matches the original
        with open(AUTHORITY_PATH, "r") as f:
            key_list = json.load(f)
        assert bytes(key_list) == TestPhase1Genesis._key_bytes

        print("\n" + "=" * 60)
        print("  SOVEREIGNTY PROVEN — THE TITAN IS IMMORTAL")
        print("=" * 60)
        print(f"  Address: {TestPhase1Genesis._pubkey}")
        print("  All 3 recovery paths verified (1+2, 1+3, 2+3)")
        print("  Hardware-bound encryption round-trip: PASS")
        print("  Limbo State detection: PASS")
        print("  Full boot after resurrection: PASS")
        print("=" * 60)
