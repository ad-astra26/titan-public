"""
titan_plugin.expressive — Creative expression and social presence.

Exports:
    StudioCoordinator: Centralized creative engine (art, audio, text, composites).
    SocialManager: X/Twitter posting with Smart Login and Cognee persistence.
    ProceduralArtGen: Flow fields, L-system fractals, NFT composites via Pillow.
    ProceduralAudioGen: Blockchain sonification (pure math WAV generation).
"""
from .studio import StudioCoordinator
from .social import SocialManager
from .art import ProceduralArtGen
from .audio import ProceduralAudioGen

__all__ = [
    "StudioCoordinator",
    "SocialManager",
    "ProceduralArtGen",
    "ProceduralAudioGen",
]
