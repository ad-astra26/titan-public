"""Self-signed TLS for the Console Agent (AG-TLS / AD-9, RFP_titan_app_pairing_modes §1.2a).

Stdlib-only at RUNTIME (`ssl` + `hashlib`); cert MINTING shells out to `openssl`
(present on every Linux box; off the agent hot path — runs once, at install or on
first boot). The phone pins `sha256(DER(cert))`, carried in the pairing QR as
`server_tls_pin`, so the channel is confidential + forward-secret (TLS 1.3 ECDHE) +
MITM-proof on a BARE IP with no domain or CA — using maintained platform crypto
(OpenSSL on the box, Conscrypt on the phone), not vendored primitives.

Sole-writer (AG7): ~/.titan/console_tls_cert.pem, ~/.titan/console_tls_key.pem.
"""
from __future__ import annotations

import hashlib
import os
import ssl
import subprocess
from pathlib import Path

CERT_NAME = "console_tls_cert.pem"
KEY_NAME = "console_tls_key.pem"


def cert_path(titan_dir: Path) -> Path:
    return titan_dir / CERT_NAME


def key_path(titan_dir: Path) -> Path:
    return titan_dir / KEY_NAME


def _gen_self_signed(cert: Path, key: Path) -> None:
    """Mint a long-lived self-signed cert via openssl (one-time). Raises on failure
    (CalledProcessError / FileNotFoundError) so the caller can fall back to plain HTTP
    with a loud warning rather than silently serving an un-pinnable channel."""
    cert.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048",
         "-keyout", str(key), "-out", str(cert),
         "-days", "3650", "-nodes", "-subj", "/CN=titan-console"],
        check=True, capture_output=True,
    )
    # key 0600 (secret), cert 0644 (public). dir already 0700 by the token step.
    try:
        os.chmod(key, 0o600)
        os.chmod(cert, 0o644)
    except OSError:
        pass


def ensure_console_tls(titan_dir: Path) -> tuple[Path, Path]:
    """Return (cert, key); mint on first use if absent. The agent calls this at boot
    so TLS works even when the installer did not run (dev / out-of-band)."""
    cert, key = cert_path(titan_dir), key_path(titan_dir)
    if not (cert.exists() and key.exists()):
        _gen_self_signed(cert, key)
    return cert, key


def cert_pin(cert: Path) -> str:
    """sha256 hex of the cert's DER — the value the app pins. MUST equal the Kotlin
    side's sha256(X509Certificate.getEncoded()); both hash the leaf DER bytes."""
    der = ssl.PEM_cert_to_DER_cert(cert.read_text())
    return hashlib.sha256(der).hexdigest()


def server_ssl_context(cert: Path, key: Path) -> ssl.SSLContext:
    """A TLS-server context for the Console Agent. TLS 1.2 floor (1.3 preferred) →
    ECDHE ciphers → forward secrecy."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx
