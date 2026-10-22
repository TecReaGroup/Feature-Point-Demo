"""Validated application settings."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class CameraSettings:
    index: int = 0
    backend: str = "dshow"
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass(frozen=True, slots=True)
class RecognitionSettings:
    device: str = "cuda"
    max_keypoints: int = 1024
    image_size: int = 768
    min_inliers: int = 12
    min_inlier_ratio: float = 0.45
    ransac_threshold: float = 4.0


@dataclass(frozen=True, slots=True)
class Settings:
    camera: CameraSettings
    recognition: RecognitionSettings
    log_level: str

    @classmethod
    def load(cls) -> "Settings":
        """Read TOML and reject invalid settings at startup."""
        with (PROJECT_ROOT / "config/config.toml").open("rb") as stream:
            configuration = tomllib.load(stream)
        camera = CameraSettings(**configuration.get("camera", {}))
        recognition = RecognitionSettings(**configuration.get("recognition", {}))
        level = configuration.get("logging", {}).get("level", "INFO")
        if level not in ("DEBUG", "PERF", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError("Invalid logging.level")
        if camera.backend not in ("dshow", "msmf", "auto"):
            raise ValueError("camera.backend must be dshow, msmf or auto")
        if type(camera.index) is not int or camera.index < 0:
            raise ValueError("camera.index must be a nonnegative integer")
        for name, value in (("width", camera.width), ("height", camera.height),
                            ("fps", camera.fps), ("max_keypoints", recognition.max_keypoints),
                            ("image_size", recognition.image_size),
                            ("min_inliers", recognition.min_inliers)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if recognition.device not in ("cuda", "cpu", "auto"):
            raise ValueError("recognition.device must be cuda, cpu or auto")
        if not 4 <= recognition.min_inliers <= recognition.max_keypoints:
            raise ValueError("min_inliers must be between 4 and max_keypoints")
        if recognition.image_size < 64 or not 0 < recognition.min_inlier_ratio <= 1:
            raise ValueError("Invalid image_size or min_inlier_ratio")
        if not 0 < recognition.ransac_threshold < 100:
            raise ValueError("ransac_threshold must be between 0 and 100 pixels")
        return cls(camera, recognition, level)
