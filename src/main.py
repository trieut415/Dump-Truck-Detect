import argparse
import yaml
from dumptruck_detector.pipeline.pipeline import run_pipeline, run_multi_camera_pipeline


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Dump Truck Detection Pipeline")
    parser.add_argument("--config", type=str, help="Path to YAML config file", required=True)
    parser.add_argument("--override-model", type=str, help="Override model path from config")
    parser.add_argument("--override-boundary", type=int, help="Override area boundary")
    parser.add_argument("--override-sources", nargs="+", help="Override video sources")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no display windows)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    model_path = args.override_model or config["model_path"]
    area_boundary = args.override_boundary or config["area_boundary"]
    video_sources = args.override_sources or config["video_sources"]
    headless = args.headless or config.get("headless", False)

    if len(video_sources) == 1:
        run_pipeline(
            video_url=video_sources[0],
            model_path=model_path,
            area_boundary=area_boundary,
            camera_id="cam01",
            headless=headless
        )
    else:
        run_multi_camera_pipeline(
            video_sources=video_sources,
            model_path=model_path,
            area_boundary=area_boundary,
            headless=headless
        )
