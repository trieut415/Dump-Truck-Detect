import threading
import cv2
from collections import defaultdict

from dumptruck_detector.pipeline.dumptruck_detector import DumpTruckDetector
from dumptruck_detector.common.common_utils import get_logger

logger = get_logger("pipeline")

track_trails = defaultdict(list)
MAX_TRAIL_LENGTH = 20


def draw_detections(frame, detections, detector, label="dumptruck"):
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        track_id = det.track_id
        direction = detector.active_ids.get(track_id, "N/A")

        # Save centroid for trail
        track_trails[track_id].append((cx, cy))
        if len(track_trails[track_id]) > MAX_TRAIL_LENGTH:
            track_trails[track_id].pop(0)

        # Choose color
        if direction == "inbound":
            color = (0, 255, 0)
        elif direction == "outbound":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # Draw track ID label
        text = f"{label} {track_id} ({direction})"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness=2)

        # Draw trailing line
        pts = track_trails[track_id]
        for j in range(1, len(pts)):
            cv2.line(frame, pts[j - 1], pts[j], color, 2)

    # Draw counter
    cv2.putText(
        frame,
        f"Counter: {detector.counter}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA
    )


def run_pipeline(video_url, model_path, area_boundary, camera_id, headless=False):
    detector = DumpTruckDetector(
        model_path=model_path,
        area_boundary=area_boundary,
        camera_id=camera_id
    )

    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        logger.warning(f"[{camera_id}] Failed to open video stream: {video_url}")
        return

    window_name = f"Camera {camera_id}"
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"[{camera_id}] Stream ended or read failed.")
            break

        detections = detector.process_frame(frame)
        logger.info(
            f"[{camera_id}] Frame processed. "
            f"Detections: {[(d.track_id, detector.active_ids.get(d.track_id)) for d in detections]}"
        )

        if not headless:
            draw_detections(frame, detections, detector)
            cv2.line(frame, (area_boundary, 0), (area_boundary, frame.shape[0]), (255, 255, 0), 2)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not headless:
        cv2.destroyWindow(window_name)


def run_multi_camera_pipeline(video_sources, model_path, area_boundary=480, headless=False):
    threads = []
    for idx, video_url in enumerate(video_sources):
        camera_id = f"cam{idx+1:02d}"
        logger.info(f"Starting pipeline for {camera_id} - {video_url}")
        thread = threading.Thread(
            target=run_pipeline,
            args=(video_url, model_path, area_boundary, camera_id, headless),
            daemon=True
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if not headless:
        cv2.destroyAllWindows()
