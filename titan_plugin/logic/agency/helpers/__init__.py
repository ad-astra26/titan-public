"""
titan_plugin/logic/agency/helpers/ — Helper implementations (Step 7.5).

Each helper wraps an existing Titan capability and implements the BaseHelper
protocol for registration in the HelperRegistry.

Available helpers:
  - web_search: SearXNG + Crawl4AI web research
  - infra_inspect: System metrics + log reader
  - social_post: X/Twitter social interaction
  - art_generate: Self-directed art generation
  - audio_generate: Trinity/blockchain sonification
  - coding_sandbox: AST-whitelisted isolated Python execution
  - code_knowledge: Source code introspection (6 modes)
"""
from .web_search import WebSearchHelper
from .infra_inspect import InfraInspectHelper
from .social_post import SocialPostHelper
from .art_generate import ArtGenerateHelper
from .audio_generate import AudioGenerateHelper
from .coding_sandbox import CodingSandboxHelper

__all__ = [
    "WebSearchHelper",
    "InfraInspectHelper",
    "SocialPostHelper",
    "ArtGenerateHelper",
    "AudioGenerateHelper",
    "CodingSandboxHelper",
]
