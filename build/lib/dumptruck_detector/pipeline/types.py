from dataclasses import dataclass
from typing import List

@dataclass
class Detection:
    track_id: int
    class_id: int
    bbox: List[float]
