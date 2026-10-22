"""ALIKED descriptors and LightGlue geometric object localization."""

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from lightglue import ALIKED, LightGlue

from feature_point.config import RecognitionSettings
from feature_point.recognition.geometry import Localization, localize_correspondences

LOGGER = logging.getLogger(__name__)
EXTRACTOR_MODEL = "aliked-n16rot"
REFERENCE_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)


@dataclass(frozen=True, slots=True)
class ReferenceView:
    angle: int
    features: dict
    original_points: np.ndarray


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
        LOGGER.info("Loading %s + LightGlue on %s", EXTRACTOR_MODEL, self.device_name)
        self.extractor = ALIKED(
            model_name=EXTRACTOR_MODEL,
            max_num_keypoints=settings.max_keypoints,
            detection_threshold=settings.detection_threshold,
        ).eval().to(device)
        self.matcher = LightGlue(features="aliked").eval().to(device)
        self.reference_views: list[ReferenceView] = []
        self.preferred_view = 0
        self.reference_shape = (0, 0)
        self.reference_bounds = (0, 0, 0, 0)
        self.reference_points = np.empty((0, 2), dtype=np.float32)

    def _extract(self, frame: np.ndarray, long_edge: int) -> dict:
        """Extract features with coordinates restored to the original image."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device).float() / 255.0
        return self.extractor.extract(image, resize=long_edge)

    def _extract_reference_view(
        self, frame: np.ndarray, bounds: tuple[int, int, int, int], angle: int,
    ) -> ReferenceView:
        """Extract rotated descriptors while retaining original-frame geometry."""
        height, width = frame.shape[:2]
        rotation = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
        cosine, sine = abs(rotation[0, 0]), abs(rotation[0, 1])
        rotated_width = math.ceil(round((width - 1) * cosine + (height - 1) * sine, 8)) + 1
        rotated_height = math.ceil(round((height - 1) * cosine + (width - 1) * sine, 8)) + 1
        rotation[:, 2] += np.float64([
            (rotated_width - width) / 2, (rotated_height - height) / 2,
        ])
        rotated = cv2.warpAffine(frame, rotation, (rotated_width, rotated_height))
        # Expanded rotation bounds must not shrink the object's extraction scale.
        long_edge = round(self.settings.image_size * max(rotated_width, rotated_height)
                          / max(width, height))
        features = self._extract(rotated, long_edge)
        points = features["keypoints"][0].cpu().numpy()
        inverse = cv2.invertAffineTransform(rotation)
        original = (points @ inverse[:, :2].T + inverse[:, 2]).astype(np.float32)
        x1, y1, x2, y2 = bounds
        inside = ((original[:, 0] >= x1) & (original[:, 0] < x2)
                  & (original[:, 1] >= y1) & (original[:, 1] < y2))
        selected = torch.from_numpy(inside).to(self.device)
        # LightGlue uses rotated coordinates; RANSAC uses original coordinates.
        features = {
            "keypoints": features["keypoints"][:, selected],
            "descriptors": features["descriptors"][:, selected],
            "image_size": features["image_size"],
        }
        return ReferenceView(angle, features, original[inside])

    @torch.inference_mode()
    def set_reference(self, selection: tuple) -> int:
        """Cache independent rotated views with a shared original reference boundary."""
        self.reference_views = []
        self.preferred_view = 0
        frame, bounds = selection
        x1, y1, x2, y2 = bounds
        preview_points = []
        maximum_count = 0
        for angle in REFERENCE_ANGLES:
            view = self._extract_reference_view(frame, bounds, angle)
            count = len(view.original_points)
            maximum_count = max(maximum_count, count)
            preview_points.append(view.original_points - np.float32([x1, y1]))
            if count >= self.settings.min_inliers:
                self.reference_views.append(view)
            LOGGER.info("Reference angle=%d keypoints=%d accepted=%s",
                        angle, count, count >= self.settings.min_inliers)
        merged_points = np.concatenate(preview_points)
        _, unique_indices = np.unique(np.rint(merged_points), axis=0, return_index=True)
        self.reference_points = merged_points[unique_indices]
        if not self.reference_views:
            raise ValueError(
                f"多角度参考中单个方向最多仅有 {maximum_count} 个特征点，"
                f"至少需要 {self.settings.min_inliers} 个，请扩大选区。"
            )
        self.reference_shape = (y2 - y1, x2 - x1)
        self.reference_bounds = bounds
        LOGGER.info("Reference accepted size=%s views=%d preview_points=%d",
                    self.reference_shape, len(self.reference_views), len(self.reference_points))
        return len(self.reference_points)

    @torch.inference_mode()
    def locate(self, frame: np.ndarray) -> Localization:
        """Try the last successful view first, then reacquire across remaining angles."""
        scene = self._extract(frame, self.settings.image_size)
        if scene["keypoints"].shape[1] < self.settings.min_inliers:
            return Localization(None, reason="当前画面特征不足")
        order = [self.preferred_view] + [
            index for index in range(len(self.reference_views)) if index != self.preferred_view
        ]
        strongest_rejection = Localization(None, reason="所有参考方向均未匹配")
        for index in order:
            view = self.reference_views[index]
            localization = self._locate_view(view, scene, frame.shape[:2])
            if localization.polygon is not None:
                if index != self.preferred_view:
                    LOGGER.info("Reference direction switched angle=%d matches=%d inliers=%d",
                                view.angle, localization.matches, localization.inliers)
                self.preferred_view = index
                return localization
            if (localization.inliers, localization.matches) >= (
                strongest_rejection.inliers, strongest_rejection.matches,
            ):
                strongest_rejection = localization
        return strongest_rejection

    def _locate_view(
        self, view: ReferenceView, scene: dict, frame_shape: tuple[int, int],
    ) -> Localization:
        """Verify one orientation using correspondences in original reference coordinates."""
        correspondence = self.matcher({"image0": view.features, "image1": scene})
        pairs = correspondence["matches"][0]
        count = len(pairs)
        if count < self.settings.min_inliers:
            return Localization(None, count, reason=f"匹配不足：{count}/{self.settings.min_inliers}")
        source = view.original_points[pairs[:, 0].cpu().numpy()]
        target = scene["keypoints"][0][pairs[:, 1]].cpu().numpy()
        scores = correspondence["scores"][0].cpu().numpy()
        return localize_correspondences(
            source, target, scores, self.reference_bounds, frame_shape, self.settings,
        )
