import threading
import cv2
from dumptruck_detector.pipeline.dumptruck_detector import DumpTruckDetector

def draw_detections(frame, detections, detector, label="dumptruck"):
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        direction = detector.active_ids.get(det.track_id, "N/A")

        # Color by direction
        if direction == "inbound":
            color = (0, 255, 0)  # Green
        elif direction == "outbound":
            color = (0, 0, 255)  # Red
        else:
            color = (200, 200, 200)  # Gray (unknown)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # Draw label with track ID and direction
        text = f"{label} {det.track_id} ({direction})"
        cv2.putText(
            frame, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness=2
        )

    # Draw the current counter in top-left corner
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


def run_pipeline(video_url, model_path, area_boundary, camera_id):
    """
    Runs detection pipeline for a single video stream.
    """
    detector = DumpTruckDetector(
        model_path=model_path,
        area_boundary=area_boundary,
        camera_id=camera_id
    )

    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        print(f"[{camera_id}] Failed to open video stream: {video_url}")
        return

    window_name = f"Camera {camera_id}"
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_id}] Stream ended or read failed.")
            break

        detections = detector.process_frame(frame)
        print(f"[{camera_id}] Frame processed. Detections: {[(d.track_id, detector.active_ids.get(d.track_id)) for d in detections]}")

        # draw_detections(frame, detections, detector)
        # cv2.line(frame, (area_boundary, 0), (area_boundary, frame.shape[0]), (255, 255, 0), 2)

        # cv2.imshow(window_name, frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break


    cap.release()
    cv2.destroyWindow(window_name)


def run_multi_camera_pipeline(video_sources, model_path, area_boundary=480):
    """
    Launches multiple video pipelines in parallel using threads.
    Each camera gets a unique ID and runs its own DumpTruckDetector.
    """
    threads = []
    for idx, video_url in enumerate(video_sources):
        camera_id = f"cam{idx+1:02d}"
        thread = threading.Thread(
            target=run_pipeline,
            args=(video_url, model_path, area_boundary, camera_id),
            daemon=True
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()
