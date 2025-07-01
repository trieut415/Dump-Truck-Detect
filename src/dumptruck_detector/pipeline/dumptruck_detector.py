import os
import logging
from ultralytics import YOLO
from dumptruck_detector.common.common_utils import (
    find_project_root,
    get_logger
)
from dumptruck_detector.pipeline.types import Detection


class DumpTruckDetector:
    def __init__(self, model_path, area_boundary, camera_id):
        self.model = self.load_model(model_path)
        self.counter = 0
        self.active_ids = {}  # track object IDs and their direction
        self.track_history = {}  # track_id -> previous center_x
        self.area_boundary = area_boundary
        self.alarm_on = False
        self.camera_id = camera_id

    def load_model(self, model_filename):
        base_dir = find_project_root()
        model_path = os.path.join(base_dir, "src", "dumptruck_detector", "resources", model_filename)
        return YOLO(model_path)


    def detect_and_track(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            stream=False,
            tracker="botsort.yaml",
            iou=0.3  # Default 0.45
        )[0]


        detections = []
        if not hasattr(results, "boxes") or results.boxes is None:
            return []

        boxes = results.boxes
        for i in range(len(boxes)):
            track_id = int(boxes.id[i].item()) if boxes.id is not None else None
            class_id = int(boxes.cls[i].item())
            bbox = boxes.xyxy[i].tolist()
            detections.append(Detection(track_id, class_id, bbox))

        return detections

    def classify_direction(self, track_id, bbox):
        """
        Determine direction based on whether the object crossed a vertical boundary line.
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2

        # No prior position → can't determine yet
        if track_id not in self.track_history:
            self.track_history[track_id] = center_x
            return None  # skip counting this frame

        prev_x = self.track_history[track_id]
        self.track_history[track_id] = center_x  # update history

        # Hardcoded boundary check
        if prev_x < self.area_boundary <= center_x:
            return "inbound"
        elif prev_x > self.area_boundary >= center_x:
            return "outbound"
        else:
            return None  # hasn't crossed boundary yet


    def update_counter(self, detections):
        for det in detections:
            direction = self.classify_direction(det.track_id, det.bbox)
            if det.track_id not in self.active_ids:
                self.active_ids[det.track_id] = direction
                if direction == 'inbound':
                    self.counter += 1
            else:
                prev_direction = self.active_ids[det.track_id]
                if prev_direction == 'inbound' and direction == 'outbound':
                    self.counter -= 1
                    del self.active_ids[det.track_id]
        self._check_alarm_state()

    def _check_alarm_state(self):
        if self.counter > 0 and not self.alarm_on:
            self.trigger_alarm(True)
        elif self.counter == 0 and self.alarm_on:
            self.trigger_alarm(False)

    def trigger_alarm(self, state: bool):
        self.alarm_on = state
        # Send TCP/IP camera command here
        print(f"Alarm {'ON' if state else 'OFF'}")

    def process_frame(self, frame):
        detections = self.detect_and_track(frame)
        self.update_counter(detections)
        return detections
