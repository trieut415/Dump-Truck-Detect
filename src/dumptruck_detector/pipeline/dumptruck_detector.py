import os
from ultralytics import YOLO
from dumptruck_detector.common.common_utils import (
    find_project_root,
    get_logger
)
from dumptruck_detector.pipeline.types import Detection


class DumpTruckDetector:
    def __init__(self, model_path, area_boundary, camera_id):
        self.logger = get_logger(f"detector:{camera_id}")
        self.model = self.load_model(model_path)
        self.counter = 0
        self.active_ids = {}  # track object IDs and their direction
        self.track_history = {}  # track_id -> previous center_x
        self.classified = set() 
        self.crossing_counts = {}
        self.area_boundary = area_boundary
        self.alarm_on = False
        self.camera_id = camera_id
        self.start_side = {}

    def load_model(self, model_filename):
        base_dir = find_project_root()
        model_path = os.path.join(base_dir, "src", "dumptruck_detector", "resources", model_filename)
        self.logger.info(f"Loading model from {model_path}")
        return YOLO(model_path)

    def detect_and_track(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            stream=False,
            tracker="botsort.yaml",
            iou=0.3
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
        x1, _, x2, _ = bbox
        cx  = (x1 + x2) / 2
        bnd = self.area_boundary

        # ── side bookkeeping ─────────────────────────────────────────────────────
        box_left  = x2 < bnd
        box_right = x1 > bnd
        box_span  = not (box_left or box_right)

        if track_id not in self.track_history:          # first sighting
            self.track_history[track_id]  = cx
            self.crossing_counts[track_id] = 0
            self.start_side[track_id]      = box_right  # True = started right
            return None

        self.track_history[track_id] = cx
        started_on_right = self.start_side[track_id]

        # opposite side (incl. touching the line)
        opposite = (box_left or box_span) if started_on_right else (box_right or box_span)

        # debounce ---------------------------------------------------------------
        self.crossing_counts[track_id] = (
            self.crossing_counts[track_id] + 1 if opposite else 0
        )

        if self.crossing_counts[track_id] >= 3:
            self.crossing_counts[track_id] = 0
            self.start_side[track_id]      = not started_on_right
        
        return "outbound" if started_on_right else "inbound"





    def update_counter(self, detections):
        for det in detections:
            direction = self.classify_direction(det.track_id, det.bbox)

            # ignore frames where the debounce has not yet settled
            if direction is None:
                continue

            if det.track_id not in self.active_ids:
                # first definitive direction for this ID
                self.active_ids[det.track_id] = direction
                if direction == "inbound":
                    self.counter += 1
                    self.logger.info(f"[{self.camera_id}] Inbound count incremented → {self.counter}")
            else:
                prev_direction = self.active_ids[det.track_id]
                if prev_direction != direction:
                    # truck turned around – adjust the tally
                    if prev_direction == "inbound" and direction == "outbound":
                        self.counter -= 1
                    elif prev_direction == "outbound" and direction == "inbound":
                        self.counter += 1
                    self.active_ids[det.track_id] = direction  # update to new state

        self._check_alarm_state()


    def _check_alarm_state(self):
        if self.counter > 0 and not self.alarm_on:
            self.trigger_alarm(True)
        elif self.counter == 0 and self.alarm_on:
            self.trigger_alarm(False)

    def trigger_alarm(self, state: bool):
        self.alarm_on = state
        self.logger.info(f"[{self.camera_id}] Alarm {'ON' if state else 'OFF'}")

    def process_frame(self, frame):
        detections = self.detect_and_track(frame)
        self.update_counter(detections)
        return detections
