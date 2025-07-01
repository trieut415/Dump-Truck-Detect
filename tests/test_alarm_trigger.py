import pytest
import numpy as np
from dumptruck_detector.pipeline.dumptruck_detector import DumpTruckDetector
from dumptruck_detector.pipeline.types import Detection

def test_alarm_toggle(dummy_detector):
    dummy_detector.trigger_alarm(False)
    assert not dummy_detector.alarm_on
    dummy_detector.trigger_alarm(True)
    assert dummy_detector.alarm_on