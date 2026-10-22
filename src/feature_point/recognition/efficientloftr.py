"""EfficientLoFTR image-pair matching through the upstream-supported Transformers port."""

import logging
from dataclasses import replace

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from feature_point.config import PROJECT_ROOT, RecognitionSettings
from feature_point.recognition.geometry import Localization, localize_correspondences

LOGGER = logging.getLogger(__name__)
MODEL_ID = "zju-community/efficientloftr"
MATCH_THRESHOLD = 0.2
IMAGE_DIVISOR = 32


class EfficientLoFTRMatcher:
    """Own pair-dependent inference and retain only matches inside the selected object."""

    def __init__(self, settings: RecognitionSettings) -> None:
        self.settings = settings
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请安装支持 CUDA 的 PyTorch 或设置 recognition.device = cpu。")
        self.device = torch.device(device)
        self.device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        cache = PROJECT_ROOT / "data/model/efficientloftr"
        LOGGER.info("Loading %s on %s cache=%s", MODEL_ID, self.device_name, cache)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=str(cache))
        self.model = AutoModel.from_pretrained(MODEL_ID, cache_dir=str(cache)).eval().to(self.device)
        self.reference_points = np.empty((0, 2), dtype=np.float32)
        self.reference_image = None
        self.reference_bounds = (0, 0, 0, 0)

    def set_reference(self, selection: tuple) -> int:
        """Store the reference frame; semi-dense correspondences require a live image pair."""
        frame, self.reference_bounds = selection
        self.reference_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.reference_points = np.empty((0, 2), dtype=np.float32)
        LOGGER.info("EfficientLoFTR reference accepted bounds=%s", self.reference_bounds)
        return 0

    @torch.inference_mode()
    def locate(self, frame: np.ndarray) -> Localization:
        """Match full images, restore pixel coordinates, then verify only the reference ROI."""
        height, width = self.reference_image.shape[:2]
        scale = self.settings.image_size / max(height, width)
        size = {
            "height": max(IMAGE_DIVISOR, round(height * scale / IMAGE_DIVISOR) * IMAGE_DIVISOR),
            "width": max(IMAGE_DIVISOR, round(width * scale / IMAGE_DIVISOR) * IMAGE_DIVISOR),
        }
        images = [self.reference_image, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]
        inputs = self.processor(images, size=size, return_tensors="pt").to(self.device)
        prediction = self.model(**inputs)
        matches = self.processor.post_process_keypoint_matching(
            prediction, [[(height, width), frame.shape[:2]]], threshold=MATCH_THRESHOLD,
        )[0]
        source = matches["keypoints0"].float().cpu().numpy()
        target = matches["keypoints1"].float().cpu().numpy()
        scores = matches["matching_scores"].float().cpu().numpy()
        x1, y1, x2, y2 = self.reference_bounds
        inside = ((source[:, 0] >= x1) & (source[:, 0] < x2)
                  & (source[:, 1] >= y1) & (source[:, 1] < y2))
        source, target, scores = source[inside], target[inside], scores[inside]
        self.reference_points = source - np.float32([x1, y1])
        localization = localize_correspondences(
            source, target, scores, self.reference_bounds, frame.shape[:2], self.settings,
        )
        return replace(localization, reference_points=self.reference_points.copy())
