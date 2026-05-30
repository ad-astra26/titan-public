#!/usr/bin/env bash
# restore_test_netjail.sh — W1.5 live-test network isolation (PRIMARY guard).
#
# The resurrection live test boots a RESTORED T1 on a fresh box to compare it
# against the LIVING mainnet T1. A restored T1 holds the SAME keypair, so if it
# could reach the public internet it would race the real T1: double on-chain
# writes, real SOL spend, duplicate X posts, Arweave backup publishes. This
# script drops all NEW outbound traffic to the public internet while keeping:
#
#   - loopback (the API / TC² talk to themselves)
#   - ESTABLISHED/RELATED replies (so YOUR inbound SSH + API calls work)
#   - the DigitalOcean VPC CIDR (so VPC-side testing + the VPC DNS resolver work)
#
# Net effect: the restored Titan fires every loop realistically, but its
# sovereign actions (Solana RPC, Arweave, X) can never LEAVE the box. This is
# the primary isolation; `setup_titan restore --verify-only` is the in-code
# backstop. Defence in depth (RFP decision #17).
#
# Usage:
#   sudo bash restore_test_netjail.sh up      [VPC_CIDR]   # engage  (default 10.0.0.0/8)
#   sudo bash restore_test_netjail.sh down                 # release
#   sudo bash restore_test_netjail.sh status               # print the ruleset
#
# Idempotent: 'up' tears down any prior jail table first. Requires nftables
# (modern Ubuntu/DO default). Must run as root.
set -euo pipefail

TABLE="titan_restore_jail"
VPC_CIDR="${2:-10.0.0.0/8}"     # DO VPC space; override per-droplet if narrower
ACTION="${1:-status}"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: must run as root (sudo)." >&2
    exit 1
  fi
}

require_nft() {
  if ! command -v nft >/dev/null 2>&1; then
    echo "ERROR: nftables (nft) not found. Install with: apt-get install -y nftables" >&2
    exit 1
  fi
}

down() {
  # Delete both families' tables if present (no error if absent).
  nft list table inet "${TABLE}" >/dev/null 2>&1 && nft delete table inet "${TABLE}" || true
  echo "[netjail] released — egress unrestricted."
}

up() {
  down   # idempotent: clear any prior jail first

  # Resolve the active DNS server so the Titan can still resolve names (the
  # subsequent public connection is what gets dropped). Fall back to the VPC.
  local dns
  dns="$(awk '/^nameserver/ {print $2; exit}' /etc/resolv.conf 2>/dev/null || true)"

  nft -f - <<EOF
table inet ${TABLE} {
  chain output {
    type filter hook output priority 0; policy drop;

    # 1. loopback — API + TC² self-talk
    oif "lo" accept

    # 2. replies to inbound connections (your SSH session, API calls)
    ct state established,related accept

    # 3. intra-VPC traffic (VPC testing + VPC resolver)
    ip daddr ${VPC_CIDR} accept

    # 4. DNS resolution (names resolve; the public connection still drops)
$( [[ -n "${dns}" ]] && echo "    ip daddr ${dns} udp dport 53 accept" )
$( [[ -n "${dns}" ]] && echo "    ip daddr ${dns} tcp dport 53 accept" )

    # 5. everything else to the public internet (IPv4) — DROPPED
    #    (policy drop above already covers it; explicit for the log/audit)
    ip daddr 0.0.0.0/0 drop

    # 6. all IPv6 egress — DROPPED (no VPC IPv6 in this test topology)
    ip6 daddr ::/0 drop
  }
}
EOF

  echo "[netjail] ENGAGED — public egress dropped (VPC ${VPC_CIDR} + lo + established allowed)."
  echo "[netjail] DNS resolver allowed: ${dns:-none}"
  echo
  status
}

status() {
  if nft list table inet "${TABLE}" >/dev/null 2>&1; then
    echo "[netjail] ACTIVE ruleset:"
    nft list table inet "${TABLE}"
  else
    echo "[netjail] inactive (no ${TABLE} table)."
  fi
}

case "${ACTION}" in
  up)     require_root; require_nft; up ;;
  down)   require_root; require_nft; down ;;
  status) require_nft; status ;;
  *)      echo "usage: sudo bash $0 {up [VPC_CIDR]|down|status}" >&2; exit 2 ;;
esac
