# Dumptruck Detector
<img src="src/eartag_jetson/resources/demo.jpg" alt="dumptruck tracking" width="100%">
This repository contains a data pipeline for dumptruck detection and tracking.

# Get Started

## 2. Get Started

### Clone the Repository
```bash
git clone https://github.com/trieut415/Dump-Truck-Detect.git
```

### Installation
Run the following script to install all necessary dependencies:
```bash
cd Dump-Truck-Detect
python3 -m venv dumptruck_env
source dumptruck_env/bin/activate
pip install -r requirements.txt
```

Activate the virtual environment:
```bash
source labby-eartag/bin/activate
```

## Command-Line Arguments

The pipeline accepts configuration via a YAML file. The default configuration is stored in [config.yaml](). You may also specify your own yaml config file:

Use the `--config` flag to specify the path.

```bash
python src/main.py --config config.yaml
```
---
To use CLI arguments instead:
```bash
python src/main.py \
  --config config.yaml \
  --model-path path/to/model \
  --video-sources rtsp://localhost:8554/cam1 rtsp://localhost:8554/cam2 \
  --area-boundary 500 \
  --headless
```
If you don't pass an argument explicitly, values from config.yaml will be used as fallback. Below is a guide of all possible CLI args.

| Argument          | Type   | Description                                                                   |
| ----------------- | ------ | ----------------------------------------------------------------------------- |
| `--config`        | `str`  | **(Required)** Path to the YAML configuration file.                           |
| `--model-path`    | `str`  | Override model path specified in the YAML. Path to the YOLO `.pt` file.       |
| `--video-sources` | `list` | Override video sources. Accepts a list of file paths or RTSP stream URLs.     |
| `--area-boundary` | `int`  | Optional. Sets the x-coordinate used to classify direction (default: 480).    |
| `--headless`      | `flag` | Optional. Run in headless mode (no display). Useful for servers or logs.      |
| `--camera-ids`    | `list` | Optional. Custom camera IDs for multi-stream setup (must match source count). |


