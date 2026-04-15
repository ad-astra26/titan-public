"""
titan_plugin.core — Core subsystems of the Titan Sovereign Stack.

Exports:
    TieredMemoryGraph: Cognee-backed dual-tier memory with neuroplasticity.
    MetabolismController: SOL balance → energy states and growth metrics.
    HybridNetworkClient: Solana RPC with fallback, priority fees, Jito bundles.
    SovereignSoul: On-chain identity via Metaplex Core NFTs.
"""
from .memory import TieredMemoryGraph
from .metabolism import MetabolismController
from .network import HybridNetworkClient
from .soul import SovereignSoul

__all__ = [
    "TieredMemoryGraph",
    "MetabolismController",
    "HybridNetworkClient",
    "SovereignSoul",
]
