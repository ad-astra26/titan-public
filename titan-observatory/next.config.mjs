/** @type {import('next').NextConfig} */
const nextConfig = {
  // distDir is overridable via NEXT_DIST_DIR so the deploy script can build
  // into a side-by-side directory (.next-build/) while the running server
  // keeps serving from the current .next/ — atomic swap at the end gives
  // ~3s of downtime instead of ~5min. See scripts/observatory_deploy.sh.
  distDir: process.env.NEXT_DIST_DIR || '.next',
  images: {
    // Titan API origins serve creative-work images:
    //   prod  https://iamtitan.tech/...   (via nginx)
    //   dev   http://localhost:7777/...   (T1)
    //   LAN   http://10.135.0.6:7777      (T2)
    //   LAN   http://10.135.0.6:7778      (T3)
    // remotePatterns is the Next.js 14 mechanism to allowlist origins for the
    // built-in image optimizer (/_next/image?url=...&w=...&q=...). Optimizer
    // generates WebP/AVIF on demand, caches in .next/cache/images/, serves
    // appropriately-sized variants to clients. Per rFP §1.4.A.
    remotePatterns: [
      { protocol: 'https', hostname: '**' },
      { protocol: 'http',  hostname: 'localhost' },
      { protocol: 'http',  hostname: '127.0.0.1' },
      { protocol: 'http',  hostname: '10.135.0.6' },
    ],
    // Default formats: try AVIF first (smallest), fallback to WebP, fallback to original.
    formats: ['image/avif', 'image/webp'],
  },
  transpilePackages: ['three'],
  // Edge rewrites so the browser can use same-origin relative URLs without
  // tripping CORS when running on localhost dev. Mirrors prod nginx routing.
  //
  //   • Prod (iamtitan.tech): nginx intercepts /v4/* + /status* + /health +
  //     /t2/* + /t3/* + /api/chat* BEFORE the request reaches next-server,
  //     forwarding to the appropriate FastAPI backend. These rewrites only
  //     fire for paths nginx forwards to next-server, so prod is unaffected.
  //
  //   • Dev (localhost:3000): there is no nginx — next-server is the origin
  //     for everything. These rewrites forward the same paths to the
  //     appropriate backend (localhost:7777 for T1, 10.135.0.6:7777 for T2,
  //     10.135.0.6:7778 for T3).
  //
  // Pair this with lib/api.ts _resolveApiBase() which emits same-origin
  // relative URLs from the browser. Closes the long-standing CORS pain
  // (localhost fetch → iamtitan.tech → preflight reject).
  async rewrites() {
    return [
      // T1 — local backend
      { source: '/v6/:path*',      destination: 'http://localhost:7777/v6/:path*' },
      { source: '/v4/:path*',      destination: 'http://localhost:7777/v4/:path*' },
      { source: '/status',         destination: 'http://localhost:7777/status' },
      { source: '/status/:path*',  destination: 'http://localhost:7777/status/:path*' },
      { source: '/health',         destination: 'http://localhost:7777/health' },
      // Media (autonomous art / audio under /media/data/studio_exports/...).
      // The same-origin API_BASE change (lib/api.ts) made image src relative
      // (`/media/...`); the Next.js image optimizer fetches the url server-side
      // and must reach the backend. nginx routes /media in prod; this rewrite
      // covers the next-server optimizer + dev. Without it, the optimizer
      // resolved `/media/...` against its own origin and returned null →
      // "isn't a valid image" → blank Autonomous Art tiles in /feed.
      { source: '/media/:path*',   destination: 'http://localhost:7777/media/:path*' },
      // Chat — /api/chat + /api/chat/stream. The /api/v4-cached + /api/page
      // + /api/bff-metrics paths above are Next.js Route Handlers and take
      // precedence over these rewrites (Route Handlers match before rewrites).
      { source: '/api/chat',           destination: 'http://localhost:7777/api/chat' },
      { source: '/api/chat/:path*',    destination: 'http://localhost:7777/api/chat/:path*' },
      // T2 / T3 — direct VPS LAN backends, stripping the /t2 + /t3 prefix
      // since the destination is the per-Titan FastAPI root (no prefix on
      // the remote side; prefix exists only for routing on the public host).
      { source: '/t2',             destination: 'http://10.135.0.6:7777/' },
      { source: '/t2/:path*',      destination: 'http://10.135.0.6:7777/:path*' },
      { source: '/t3',             destination: 'http://10.135.0.6:7778/' },
      { source: '/t3/:path*',      destination: 'http://10.135.0.6:7778/:path*' },
    ];
  },
};

export default nextConfig;
