"""Phase 6 — concrete `TruthOraclePlug` implementations (D-SPEC-PHASE6).

Each module here wraps an existing Titan subsystem as a TruthOraclePlug:

- ``coding_sandbox_oracle`` — wraps ``CodingSandboxHelper``
  (titan_hcl/logic/agency/helpers/coding_sandbox.py); FREE cost class.
- ``solana_rpc_oracle`` — wraps the Helius RPC connection (core/network.py);
  METERED (helius_rpc per INV-Syn-13).
- ``web_api_oracle`` — wraps ``knowledge_router`` with a verify-claim mode;
  METERED (web_api).
- ``x_oracle`` — wraps ``SocialXGateway`` read methods; METERED (x_api).

All four plugs satisfy the protocol from
``titan_hcl/synthesis/plugs.py:TruthOraclePlug`` and anchor their verdicts
through ``OracleRouter`` (P6.F) per INV-Syn-12 routing.
"""
