#!/usr/bin/env bash
# rFP_x_voice_enrichment §4.6 — add the /ar/ Arweave shortener route to
# nginx so PROOF_DAY (and future Arweave-backed posts) can use compact
# iamtitan.tech/ar/<tx_sig> URLs that 302 to arweave.net/<tx_sig>.
#
# This is a one-shot config change. Idempotent: re-running it is safe.
# Requires sudo (modifies /etc/nginx/sites-enabled/iamtitan.tech and
# triggers `nginx -t && systemctl reload nginx`).
#
#   bash scripts/nginx_xvoice_ar_route.sh
#
set -euo pipefail

CONF=/etc/nginx/sites-enabled/iamtitan.tech

if ! sudo grep -q "location /ar/" "$CONF"; then
    echo "[nginx_xvoice_ar_route] adding /ar/ location block to $CONF"
    # Insert directly after the /tx/ block.
    sudo python3 - "$CONF" <<'PY'
import sys, re
path = sys.argv[1]
src = open(path).read()
if "location /ar/" in src:
    sys.exit(0)
new_block = '''
    # Arweave artifact shortener (rFP_x_voice_enrichment §4.6 — PROOF_DAY)
    location /ar/ {
        rewrite ^/ar/(.+)$ https://arweave.net/$1 redirect;
    }
'''
src2 = re.sub(
    r'(location /tx/ \{[^}]*\})',
    r'\1\n' + new_block,
    src, count=1,
)
if src2 == src:
    # /tx/ block missing → append before final '}' of the server block.
    src2 = re.sub(r'\}\s*$', new_block + '}\n', src)
open(path, 'w').write(src2)
PY
else
    echo "[nginx_xvoice_ar_route] /ar/ already configured"
fi

echo "[nginx_xvoice_ar_route] running nginx -t"
sudo nginx -t
echo "[nginx_xvoice_ar_route] reloading nginx"
sudo systemctl reload nginx
echo "[nginx_xvoice_ar_route] done. test with:"
echo "  curl -s -o /dev/null -w '%{http_code} -> %{redirect_url}\\n' https://iamtitan.tech/ar/test_tx_sig"
