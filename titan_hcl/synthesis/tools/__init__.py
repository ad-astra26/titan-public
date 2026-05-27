"""Phase 6 — concrete `ToolPlug` implementations (D-SPEC-PHASE6).

INV-12: the engine has ONE action / tool surface — existing tools
(events_teacher, knowledge router, coding sandbox, SocialXGateway)
folded in; no parallel tool-invocation paths surviving.

Four day-one ToolPlugs per SPEC §25.3 + arch §11.3:

- ``coding_sandbox_tool.CodingSandboxTool`` — wraps `CodingSandboxHelper`;
  doubles as `TruthOraclePlug` (P6.B). Closes arch §11.3 "sandbox unused
  by the outer self" gap by getting wired into the agno tool layer.
- ``events_teacher_tool.EventsTeacherTool`` — wraps the events_teacher
  X-event distillation surface.
- ``knowledge_tool.KnowledgeTool`` — wraps the existing knowledge_router
  + StealthSage research path; same underlying infra as the P6.D
  ``web_api`` oracle (one process, two surfaces).
- ``x_research_tool.XResearchTool`` — wraps `SocialXGateway` read/post
  paths (post-side only via SocialXGateway per
  ``feedback_social_x_gateway_post_is_sole_sanctioned_x_path``); doubles
  as `TruthOraclePlug` (P6.E `x_oracle`).

Every tool invocation emits a procedural-fork TX (arch §8.2) carrying
the ``scored_by`` field (P6.J — populated by the OracleRouter companion
verdict path) so the §A.6 ≥95% coverage gate becomes measurable.
"""
