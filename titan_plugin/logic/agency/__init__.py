"""
titan_plugin/logic/agency/ — Agency Module (Step 7.3+).

Provides the autonomous action pipeline:
  Impulse → Intent → Helper Selection → Execution → Assessment → Enrichment

Sub-modules:
  - registry.py: Helper registry with BaseHelper protocol and manifests
  - module.py: Agency orchestrator (Step 7.4)
  - assessment.py: IQL self-assessment loop (Step 7.6)
  - helpers/: Individual helper implementations (Step 7.5, 7.7)
"""
from .registry import BaseHelper, HelperRegistry
from .module import AgencyModule
from .assessment import SelfAssessment

__all__ = ["BaseHelper", "HelperRegistry", "AgencyModule", "SelfAssessment"]
