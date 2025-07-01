import pytest
import numpy as np
from dumptruck_detector.pipeline.dumptruck_detector import DumpTruckDetector
from dumptruck_detector.pipeline.types import Detection

def test_sustained_crossing_trail(dummy_detector):
    track_id = 99
    # cross at boundary=500 with 3-frame dwell
    trail = [
      (480,200),
      (510,200),  # 1
      (515,200),  # 2
      (520,200),  # 3 -> inbound
    ]
    direction = None
    for x,y in trail:
        bbox = [x-20,y-20,x+20,y+20]
        direction = dummy_detector.classify_direction(track_id, bbox)
    assert direction == "inbound"
    # and history ends at last center
    assert dummy_detector.track_history[track_id] == trail[-1][0]
