"""Shared geometric verification of image-pair correspondences."""

from dataclasses import dataclass

import cv2
import numpy as np

from feature_point.config import RecognitionSettings

MIN_QUADRILATERAL_AREA = 100.0
MIN_REFERENCE_COVERAGE = 0.02


@dataclass(frozen=True, slots=True)
class Localization:
    polygon: np.ndarray | None
    matches: int = 0
    inliers: int = 0
    reason: str = ""
    confidence: float = 0.0
    reference_points: np.ndarray | None = None


def localize_correspondences(
    source: np.ndarray, target: np.ndarray, scores: np.ndarray,
    bounds: tuple[int, int, int, int], frame_shape: tuple[int, int],
    settings: RecognitionSettings,
) -> Localization:
    """Project a reference boundary only after spatial and RANSAC verification."""
    count = len(source)
    if count < settings.min_inliers:
        return Localization(None, count, reason=f"匹配不足：{count}/{settings.min_inliers}")
    homography, mask = cv2.findHomography(
        source, target, cv2.RANSAC, settings.ransac_threshold,
        maxIters=2000, confidence=0.995,
    )
    if homography is None or mask is None or not np.isfinite(homography).all():
        return Localization(None, count, reason="无法估计稳定的几何变换")
    inliers = int(mask.sum())
    rejected = Localization(None, count, inliers, "几何验证未通过")
    if inliers < settings.min_inliers or inliers / count < settings.min_inlier_ratio:
        return rejected
    x1, y1, x2, y2 = bounds
    support = cv2.convexHull(source[mask.ravel().astype(bool)])
    if cv2.contourArea(support) < (x2 - x1) * (y2 - y1) * MIN_REFERENCE_COVERAGE:
        return rejected
    corners = np.float32([[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]])
    projected = np.column_stack((corners, np.ones(4))) @ homography.T
    denominator = projected[:, 2]
    if not (np.all(denominator > 1e-6) or np.all(denominator < -1e-6)):
        return rejected
    polygon = (projected[:, :2] / denominator[:, None]).astype(np.float32)
    frame_height, frame_width = frame_shape
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
    # Matching scores are not calibrated probabilities of object identity.
    confidence = float(np.clip(scores[mask.ravel().astype(bool)].mean() * inliers / count, 0, 1))
    return Localization(polygon, count, inliers, confidence=confidence)
