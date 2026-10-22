"""ALIKED descriptors and LightGlue geometric object localization."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from lightglue import ALIKED, LightGlue

from feature_point.config import RecognitionSettings

LOGGER = logging.getLogger(__name__)
MIN_QUADRILATERAL_AREA = 100.0
MIN_REFERENCE_COVERAGE = 0.02


@dataclass(frozen=True, slots=True)
class Localization:
    polygon: np.ndarray | None
    matches: int = 0
    inliers: int = 0


class ObjectMatcher:
    """Own models and cached reference features on the inference thread."""

    def __init__(self, settings: RecognitionSettings) -> None:
        self.settings = settings
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 不可用。请安装支持 CUDA 的 torch/torchvision，"
                "或将 config/config.toml 中 recognition.device 改为 cpu。"
            )
        self.device = torch.device(device)
        self.device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        LOGGER.info("Loading ALIKED + LightGlue on %s", self.device_name)
        self.extractor = ALIKED(max_num_keypoints=settings.max_keypoints).eval().to(device)
        self.matcher = LightGlue(features="aliked").eval().to(device)
        self.reference_features = None
        self.reference_shape = (0, 0)

    def _extract(self, frame: np.ndarray) -> dict:
        """Extract features with coordinates restored to the original image."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device).float() / 255.0
        return self.extractor.extract(image, resize=self.settings.image_size)

    @torch.inference_mode()
    def set_reference(self, crop: np.ndarray) -> int:
        """Cache reference descriptors once for each user selection."""
        self.reference_features = None
        features = self._extract(crop)
        count = features["keypoints"].shape[1]
        if count < self.settings.min_inliers:
            raise ValueError(f"参考区域仅有 {count} 个特征点，请框选更大、纹理更丰富的区域。")
        self.reference_features = features
        self.reference_shape = crop.shape[:2]
        LOGGER.info("Reference accepted size=%s keypoints=%d", self.reference_shape, count)
        return count

    @torch.inference_mode()
    def locate(self, frame: np.ndarray) -> Localization:
        """Return a quadrilateral only after spatial and RANSAC verification."""
        scene = self._extract(frame)
        if scene["keypoints"].shape[1] < self.settings.min_inliers:
            return Localization(None)
        correspondence = self.matcher({"image0": self.reference_features, "image1": scene})
        pairs = correspondence["matches"][0]
        count = len(pairs)
        if count < self.settings.min_inliers:
            return Localization(None, count)
        source = self.reference_features["keypoints"][0][pairs[:, 0]].cpu().numpy()
        target = scene["keypoints"][0][pairs[:, 1]].cpu().numpy()
        homography, mask = cv2.findHomography(
            source, target, cv2.RANSAC, self.settings.ransac_threshold,
            maxIters=2000, confidence=0.995,
        )
        if homography is None or mask is None or not np.isfinite(homography).all():
            return Localization(None, count)
        inliers = int(mask.sum())
        rejected = Localization(None, count, inliers)
        if inliers < self.settings.min_inliers or inliers / count < self.settings.min_inlier_ratio:
            return rejected
        height, width = self.reference_shape
        support = cv2.convexHull(source[mask.ravel().astype(bool)])
        if cv2.contourArea(support) < width * height * MIN_REFERENCE_COVERAGE:
            return rejected
        corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        projected = np.column_stack((corners, np.ones(4))) @ homography.T
        denominator = projected[:, 2]
        if not (np.all(denominator > 1e-6) or np.all(denominator < -1e-6)):
            return rejected
        polygon = (projected[:, :2] / denominator[:, None]).astype(np.float32)
        frame_height, frame_width = frame.shape[:2]
        if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
            return rejected
        area = cv2.contourArea(polygon)
        if not MIN_QUADRILATERAL_AREA <= area <= frame_width * frame_height * 1.5:
            return rejected
        if (polygon[:, 0].min() < -frame_width * 0.25
                or polygon[:, 0].max() > frame_width * 1.25
                or polygon[:, 1].min() < -frame_height * 0.25
                or polygon[:, 1].max() > frame_height * 1.25):
            return rejected
        return Localization(polygon, count, inliers)
