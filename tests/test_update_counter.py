import pytest
import numpy as np
from dumptruck_detector.pipeline.dumptruck_detector import DumpTruckDetector
from dumptruck_detector.pipeline.types import Detection

def test_update_counter_with_debounce(dummy_detector):
    dets = [
        Detection(track_id=10, class_id=0, bbox=[480, 100, 520, 140]),
        Detection(track_id=10, class_id=0, bbox=[510, 100, 550, 140]),
        Detection(track_id=10, class_id=0, bbox=[515, 100, 555, 140]),
        Detection(track_id=10, class_id=0, bbox=[520, 100, 560, 140]),
    ]

    for det in dets:
        dummy_detector.update_counter([det])

    assert dummy_detector.counter == 1
