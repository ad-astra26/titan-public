#!/usr/bin/env node
/**
 * irys_upload_daemon.js — Long-running Irys SDK helper for backup pipeline.
 *
 * Phase 5 chunk 5E (rFP_phase_c_enhancements §3B.2, 2026-05-19).
 *
 * Pre-5E pattern: `scripts/irys_upload.js` is spawned per call. Each spawn
 * loads @irys/sdk (~250 ms), constructs an Irys client (~500-800 ms RPC ping
 * + balance query), then exits. Across a baseline event that ships 3 tiers
 * × 1+ tarballs each plus a balance check and a fund check, that's
 * 5-10 cold spawns and 3-8 seconds of pure SDK startup overhead.
 *
 * This daemon keeps ONE Irys instance loaded and serves commands over
 * stdin/stdout in newline-delimited JSON. The Python wrapper
 * (irys_persistent_client.py) spawns one daemon per (keypair, rpc) pair,
 * reuses it across operations, and sends `shutdown` on Python exit.
 *
 * Wire protocol (JSONL — one JSON object per line):
 *   request:  {"id":"req_001","op":"upload_file","path":"/tmp/x","content_type":"...","tags":{...}}
 *   response: {"id":"req_001","status":"ok","tx_id":"...","url":"...","size":N}
 *   error:    {"id":"req_001","status":"error","message":"..."}
 *
 * Supported ops:
 *   - ready    → no-op handshake; replies {"status":"ok","ready":true}
 *   - balance  → {"status":"ok","balance_atomic":"...","balance_readable":"..."}
 *   - fund {amount_lamports} → {"status":"ok","funded":"...","target":"...","tx_id":"..."}
 *   - price {size_bytes} → {"status":"ok","price_lamports":"..."} (Irys getPrice)
 *   - upload_file {path, content_type?, tags?} → {"status":"ok","tx_id":"...","url":"...","size":N}
 *   - upload_data {data_b64, content_type?, tags?} → {"status":"ok","tx_id":"...","url":"...","size":N}
 *   - shutdown → {"status":"ok","bye":true} then process.exit(0)
 *
 * Startup args:
 *   node irys_upload_daemon.js <keypair_path> [rpc_url]
 *
 * Failure model:
 *   - Per-request errors return {"status":"error",...}; the daemon stays up.
 *   - Unrecoverable SDK errors crash the process — the Python wrapper
 *     observes EOF on stdout and respawns.
 */

const Irys = require("@irys/sdk");
const fs = require("fs");
const readline = require("readline");

const IRYS_NODE = "https://node2.irys.xyz";

async function buildIrys(keypairPath, rpcUrl) {
    const keypairData = JSON.parse(fs.readFileSync(keypairPath, "utf-8"));
    const secret = Uint8Array.from(keypairData);
    return new Irys({
        url: IRYS_NODE,
        token: "solana",
        key: secret,
        config: {
            providerUrl: rpcUrl || "https://api.mainnet-beta.solana.com",
        },
    });
}

function buildTags(contentType, extra) {
    const tags = [];
    tags.push({ name: "App-Name", value: "Titan-Sovereign-AI" });
    tags.push({ name: "App-Version", value: "6.0" });
    if (contentType) {
        tags.push({ name: "Content-Type", value: contentType });
    }
    if (extra && typeof extra === "object") {
        for (const [k, v] of Object.entries(extra)) {
            tags.push({ name: k, value: String(v) });
        }
    }
    return tags;
}

function emit(obj) {
    process.stdout.write(JSON.stringify(obj) + "\n");
}

async function handle(irys, req) {
    const op = req.op;
    if (op === "ready") {
        return { status: "ok", ready: true };
    }
    if (op === "balance") {
        const bal = await irys.getLoadedBalance();
        return {
            status: "ok",
            balance_atomic: bal.toString(),
            balance_readable: irys.utils.fromAtomic(bal).toString(),
        };
    }
    if (op === "fund") {
        const amount = parseInt(req.amount_lamports, 10);
        if (!Number.isFinite(amount) || amount <= 0) {
            return { status: "error", message: "amount_lamports must be a positive integer" };
        }
        const receipt = await irys.fund(amount);
        return {
            status: "ok",
            funded: receipt.quantity.toString(),
            target: receipt.target,
            tx_id: receipt.id,
        };
    }
    if (op === "price") {
        const size = parseInt(req.size_bytes, 10);
        if (!Number.isFinite(size) || size <= 0) {
            return { status: "error", message: "size_bytes must be a positive integer" };
        }
        const price = await irys.getPrice(size);
        return { status: "ok", price_lamports: price.toString() };
    }
    if (op === "upload_file") {
        if (!req.path || !fs.existsSync(req.path)) {
            return { status: "error", message: `path not found: ${req.path}` };
        }
        const tags = buildTags(req.content_type, req.tags);
        const receipt = await irys.uploadFile(req.path, { tags });
        const size = receipt.size || fs.statSync(req.path).size;
        return {
            status: "ok",
            tx_id: receipt.id,
            url: `https://arweave.net/${receipt.id}`,
            size,
        };
    }
    if (op === "upload_data") {
        if (!req.data_b64) {
            return { status: "error", message: "data_b64 missing" };
        }
        const data = Buffer.from(req.data_b64, "base64");
        const tags = buildTags(req.content_type, req.tags);
        const receipt = await irys.upload(data, { tags });
        return {
            status: "ok",
            tx_id: receipt.id,
            url: `https://arweave.net/${receipt.id}`,
            size: data.length,
        };
    }
    return { status: "error", message: `unknown op: ${op}` };
}

async function main() {
    const [,, keypairPath, rpcUrl] = process.argv;
    if (!keypairPath || !fs.existsSync(keypairPath)) {
        emit({ status: "error", message: "keypair_path missing or not found" });
        process.exit(2);
    }

    let irys;
    try {
        irys = await buildIrys(keypairPath, rpcUrl);
    } catch (e) {
        emit({ status: "error", message: `irys_init_failed: ${e.message || e}` });
        process.exit(3);
    }

    emit({ status: "ok", ready: true, pid: process.pid });

    const rl = readline.createInterface({
        input: process.stdin,
        terminal: false,
    });

    rl.on("line", async (line) => {
        const trimmed = line.trim();
        if (!trimmed) return;
        let req;
        try {
            req = JSON.parse(trimmed);
        } catch (e) {
            emit({ id: null, status: "error", message: `bad_json: ${e.message}` });
            return;
        }
        const id = req.id || null;
        if (req.op === "shutdown") {
            emit({ id, status: "ok", bye: true });
            process.exit(0);
        }
        try {
            const resp = await handle(irys, req);
            emit({ id, ...resp });
        } catch (e) {
            emit({ id, status: "error", message: e.message || String(e) });
        }
    });

    rl.on("close", () => {
        process.exit(0);
    });
}

main().catch((e) => {
    emit({ status: "error", message: `fatal: ${e.message || e}` });
    process.exit(1);
});
