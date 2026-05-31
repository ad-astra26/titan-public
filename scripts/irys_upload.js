#!/usr/bin/env node
/**
 * irys_upload.js — Upload files to Arweave via Irys Node 2 (Solana)
 *
 * Usage:
 *   node scripts/irys_upload.js fund <amount_lamports> <keypair_path> [rpc_url]
 *   node scripts/irys_upload.js balance <keypair_path>
 *   node scripts/irys_upload.js upload <file_path> <keypair_path> [rpc_url] [content_type] [tags_json]
 *   node scripts/irys_upload.js upload-data <keypair_path> [rpc_url] [content_type] [tags_json]
 *     (reads data from stdin)
 *
 * Output: JSON to stdout
 *   { "status": "ok", "tx_id": "...", "url": "https://arweave.net/..." }
 *   { "status": "error", "message": "..." }
 */

const Irys = require("@irys/sdk");
const fs = require("fs");
const path = require("path");

const IRYS_NODE = "https://node2.irys.xyz";

async function getIrys(keypairPath, rpcUrl) {
    const keypairData = JSON.parse(fs.readFileSync(keypairPath, "utf-8"));
    const secret = Uint8Array.from(keypairData);

    const irys = new Irys({
        url: IRYS_NODE,
        token: "solana",
        key: secret,
        config: {
            providerUrl: rpcUrl || "https://api.mainnet-beta.solana.com",
        },
    });

    return irys;
}

async function fund(amountLamports, keypairPath, rpcUrl) {
    const irys = await getIrys(keypairPath, rpcUrl);
    const receipt = await irys.fund(parseInt(amountLamports));
    return {
        status: "ok",
        funded: receipt.quantity,
        target: receipt.target,
        tx_id: receipt.id,
    };
}

async function balance(keypairPath, rpcUrl) {
    const irys = await getIrys(keypairPath, rpcUrl);
    const bal = await irys.getLoadedBalance();
    return {
        status: "ok",
        balance_atomic: bal.toString(),
        balance_readable: irys.utils.fromAtomic(bal).toString(),
    };
}

async function upload(filePath, keypairPath, rpcUrl, contentType, tagsJson) {
    const irys = await getIrys(keypairPath, rpcUrl);

    const tags = [];
    tags.push({ name: "App-Name", value: "Titan-Sovereign-AI" });
    tags.push({ name: "App-Version", value: "6.0" });

    if (contentType) {
        tags.push({ name: "Content-Type", value: contentType });
    }

    if (tagsJson) {
        const extra = JSON.parse(tagsJson);
        for (const [k, v] of Object.entries(extra)) {
            tags.push({ name: k, value: String(v) });
        }
    }

    const receipt = await irys.uploadFile(filePath, { tags });
    return {
        status: "ok",
        tx_id: receipt.id,
        url: `https://arweave.net/${receipt.id}`,
        size: receipt.size || fs.statSync(filePath).size,
    };
}

async function uploadData(keypairPath, rpcUrl, contentType, tagsJson) {
    const irys = await getIrys(keypairPath, rpcUrl);

    // Read from stdin
    const chunks = [];
    for await (const chunk of process.stdin) {
        chunks.push(chunk);
    }
    const data = Buffer.concat(chunks);

    const tags = [];
    tags.push({ name: "App-Name", value: "Titan-Sovereign-AI" });
    tags.push({ name: "App-Version", value: "6.0" });

    if (contentType) {
        tags.push({ name: "Content-Type", value: contentType });
    }

    if (tagsJson) {
        const extra = JSON.parse(tagsJson);
        for (const [k, v] of Object.entries(extra)) {
            tags.push({ name: k, value: String(v) });
        }
    }

    const receipt = await irys.upload(data, { tags });
    return {
        status: "ok",
        tx_id: receipt.id,
        url: `https://arweave.net/${receipt.id}`,
        size: data.length,
    };
}

async function main() {
    const [,, command, ...args] = process.argv;

    try {
        let result;
        switch (command) {
            case "fund":
                result = await fund(args[0], args[1], args[2]);
                break;
            case "balance":
                result = await balance(args[0], args[1]);
                break;
            case "upload":
                result = await upload(args[0], args[1], args[2], args[3], args[4]);
                break;
            case "upload-data":
                result = await uploadData(args[0], args[1], args[2], args[3]);
                break;
            default:
                result = { status: "error", message: `Unknown command: ${command}` };
        }
        console.log(JSON.stringify(result));
    } catch (err) {
        console.log(JSON.stringify({ status: "error", message: err.message || String(err) }));
        process.exit(1);
    }
}

main();
