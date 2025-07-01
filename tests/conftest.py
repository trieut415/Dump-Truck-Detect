import pytest
import numpy as np
from dumptruck_detector.pipeline.dumptruck_detector import DumpTruckDetector
from dumptruck_detector.pipeline.types import Detection

@pytest.fixture
def dummy_detector():
    return DumpTruckDetector("single_class_dumptruck.pt", 500, "test")

@pytest.fixture
def dummy_frame():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)

@pytest.fixture
def sample_detection():
    return Detection(track_id=10, class_id=0, bbox=[400, 100, 440, 140])