
"""
Minimal data classes for the project (strictly the items requested).
Bounding boxes use (x, y, w, h) in pixel coordinates.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any

BBox = Tuple[float, float, float, float]  # (x, y, w, h)

@dataclass
class Face:
    """Placeholder for face-related data. Extend later if needed."""
    pass

@dataclass
class Person:
    """Individual detected in the scene."""
    person_id: str                 # unique ID
    bbox: BBox                     # person bounding box
    mask: Optional[Any] = None     # binary mask or array-like
    face: Optional[Face] = None    # face data container

@dataclass
class Scene:
    """Scene container with people and a count metric."""
    people: Dict[str, Person] = field(default_factory=dict)

    @property
    def people_count(self) -> int:
        """Number of people currently in the scene."""
        return len(self.people)