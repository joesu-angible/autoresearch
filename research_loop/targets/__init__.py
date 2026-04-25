"""Target adapters — uniform interface so the tournament CLI is target-agnostic."""

from research_loop.targets.student_v2 import StudentV2Target
from research_loop.targets.dino_v2 import DinoV2Target

__all__ = ["StudentV2Target", "DinoV2Target"]
