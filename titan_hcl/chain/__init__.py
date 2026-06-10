"""titan_hcl.chain — ChainProvider: one lean async interface for all on-chain I/O.

RFP_chain_provider (redesign Step 1). Phase A ships the Arweave/Irys DATA plane
(put / get_to_file / get_bytes / head); the Solana trust plane (commit_memo /
read_memo / list_memos) and the funding plane (balance / fund) land in Phases
B + C and currently raise NotImplementedError.

Public surface:
    from titan_hcl.chain import ChainProvider, ArweaveChainProvider, FakeChainProvider
"""
from titan_hcl.chain.provider import (
    ChainProvider,
    ArweaveChainProvider,
    HeadStatus,
)
from titan_hcl.chain.fake import FakeChainProvider

__all__ = [
    "ChainProvider",
    "ArweaveChainProvider",
    "FakeChainProvider",
    "HeadStatus",
]
