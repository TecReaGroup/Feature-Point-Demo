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
    if inliers < settings.min_inliers:
        return Localization(None, count, inliers,
                            f"几何内点不足：{inliers}/{settings.min_inliers}")
    ratio = inliers / count
    if ratio < settings.min_inlier_ratio:
        return Localization(None, count, inliers,
                            f"内点比例不足：{ratio:.0%} < {settings.min_inlier_ratio:.0%}")
    x1, y1, x2, y2 = bounds
    support = cv2.convexHull(source[mask.ravel().astype(bool)])
    coverage = cv2.contourArea(support) / ((x2 - x1) * (y2 - y1))
    if coverage < MIN_REFERENCE_COVERAGE:
        return Localization(None, count, inliers,
                            f"内点过于集中或共线：覆盖 {coverage:.1%} < {MIN_REFERENCE_COVERAGE:.0%}")
    corners = np.float32([[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]])
    projected = np.column_stack((corners, np.ones(4))) @ homography.T
    denominator = projected[:, 2]
    if not (np.all(denominator > 1e-6) or np.all(denominator < -1e-6)):
        return Localization(None, count, inliers, "透视变换不稳定：目标边界跨越投影奇点")
    polygon = (projected[:, :2] / denominator[:, None]).astype(np.float32)
    frame_height, frame_width = frame_shape
    if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
        return Localization(None, count, inliers, "投影轮廓异常：非凸或坐标无效")
    area = cv2.contourArea(polygon)
    if not MIN_QUADRILATERAL_AREA <= area <= frame_width * frame_height * 1.5:
        return Localization(None, count, inliers, f"投影面积异常：{area:.0f} px²")
    if (polygon[:, 0].min() < -frame_width * 0.25
            or polygon[:, 0].max() > frame_width * 1.25
            or polygon[:, 1].min() < -frame_height * 0.25
            or polygon[:, 1].max() > frame_height * 1.25):
        return Localization(None, count, inliers, "投影轮廓超出画面允许范围")
    # Matching scores are not calibrated probabilities of object identity.
    confidence = float(np.clip(scores[mask.ravel().astype(bool)].mean() * inliers / count, 0, 1))
    return Localization(polygon, count, inliers, confidence=confidence)
