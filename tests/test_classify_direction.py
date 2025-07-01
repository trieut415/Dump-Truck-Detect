import pytest
import numpy as np
from dumptruck_detector.pipeline.dumptruck_detector import DumpTruckDetector
from dumptruck_detector.pipeline.types import Detection

def test_classify_direction_debounce_inbound(dummy_detector):
    bboxes = [
        [480, 100, 520, 140],  # init
        [510, 100, 550, 140],  # frame 1 across
        [515, 100, 555, 140],  # frame 2
        [520, 100, 560, 140],  # frame 3 -> should trigger inbound
    ]

    directions = []
    for box in bboxes:
        result = dummy_detector.classify_direction(1, box)
        if result:
            directions.append(result)

    assert "inbound" in directions


def test_classify_direction_debounce_outbound(dummy_detector):
    # Simulate 3 frames just crossing right→left
    bboxes = [
      [600,100,640,140],  # init
      [490,100,530,140],  # frame1 across
      [485,100,525,140],  # frame2 across
      [480,100,520,140],  # frame3 across -> now "outbound"
    ]
    result = None
    for box in bboxes:
        result = dummy_detector.classify_direction(2, box)
    assert result == "outbound"
