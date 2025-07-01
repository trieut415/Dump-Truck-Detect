import pytest
import numpy as np
from dumptruck_detector.pipeline.dumptruck_detector import DumpTruckDetector
from dumptruck_detector.pipeline.types import Detection

def test_detect_and_track_dummy(dummy_detector, dummy_frame):
    results = dummy_detector.detect_and_track(dummy_frame)
    assert isinstance(results, list)
    for det in results:
        assert hasattr(det, "track_id")
        assert hasattr(det, "bbox")
