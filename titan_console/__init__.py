"""Titan Command Center (TC²) — Console Agent.

A tiny, dependency-free (stdlib-only) ops daemon that runs as its OWN systemd
unit (`titan-console.service`), independent of the Titan process. It is the
backend for the TC² browser SPA and, crucially, stays up when the Titan is
down — so the owner can always see *why* their Titan died and act on it.

Design contract (RFP decisions #12–#15):
  - Zero third-party imports. The agent must never break because a Titan
    dependency broke. Host metrics come from /proc + os + shutil; the proxy
    to api_hcl uses urllib. (W8.)
  - OS-level powers the Titan API must NOT have: host resources, systemd
    restart/clean, journal tail, backup management, settings R/W.
  - Live cognition is PROXIED to api_hcl:7777 when up; every cognition panel
    degrades to "Titan down + why" when it isn't.

This package is the backend. The Vite/React SPA lives in `titan-console/`.
"""

__version__ = "0.1.0-alpha"
