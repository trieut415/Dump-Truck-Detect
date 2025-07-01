# src/eartag_jetson/common/common_utils.py
import os
import logging
from pathlib import Path
from typing import Sequence, Union
import shutil
from ultralytics import YOLO
import torch
from colorama import Fore, Style, init
import logging
from logging import Logger

init(autoreset=True)

from colorama import Fore, Style
import logging

class ColorFormatter(logging.Formatter):
    def format(self, record):
        level_color = {
            "DEBUG": Fore.BLUE,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Style.BRIGHT,
        }.get(record.levelname, "")

        fmt = (
            f"{Fore.WHITE}[%(asctime)s]{Style.RESET_ALL} "
            f"{level_color}[%(levelname)s]{Style.RESET_ALL} "
            f"{Fore.YELLOW}[%(processName)s]{Style.RESET_ALL} "
            f"{Fore.CYAN}%(filename)s{Style.RESET_ALL} "
            f"- %(message)s"
        )
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # clear any old handlers so you don’t double-log
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logger.addHandler(handler)

    return logger


def find_project_root(
    marker_files: Union[str, Sequence[str]] = ("pyproject.toml", "setup.py")
) -> str:
    """
    Walk upwards from this file's directory until you find one of the marker files.
    Returns the absolute path to the project root.
    Raises FileNotFoundError if none is found.
    """
    if isinstance(marker_files, str):
        marker_files = (marker_files,)
    cur = Path(__file__).parent.resolve()
    while True:
        for m in marker_files:
            if (cur / m).exists():
                return str(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(f"No project root found (checked for {marker_files})")
