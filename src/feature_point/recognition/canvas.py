"""Aspect-correct video painting and frozen-frame reference selection."""

import math

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

MIN_SELECTION_SIZE = 32

def frame_image(frame) -> QImage:
    """Return an owning RGB image safe across frame lifetimes."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class VideoCanvas(QWidget):
    """Map mouse coordinates through letterboxing into the frozen source image."""

    selected = Signal(object)
    selection_changed = Signal(bool)
    selection_rejected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.frame = None
        self.image = QImage()
        self.polygon = None
        self.selecting = False
        self.drag_start = None
        self.drag_end = None
        self.image_rect = QRectF()

    def set_frame(self, frame, polygon=None) -> None:
        """Display a frame and its own localization without temporal mismatch."""
        if self.selecting:
            return
        self.frame = frame
        self.image = frame_image(frame)
        self.polygon = polygon
        self.update()

    def begin_selection(self) -> None:
        """Freeze the displayed frame until selection completes or is cancelled."""
        if self.frame is None:
            return
        self.selecting = True
        self.polygon = None
        self.drag_start = self.drag_end = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        self.selection_changed.emit(True)
        self.update()

    def cancel_selection(self) -> None:
        """Resume preview without replacing the reference."""
        self.selecting = False
        self.drag_start = self.drag_end = None
        self.unsetCursor()
        self.selection_changed.emit(False)
        self.update()

    def _source_position(self, position: QPointF) -> QPointF:
        width, height = self.image.width(), self.image.height()
        return QPointF(
            max(0.0, min(float(width), (position.x() - self.image_rect.x())
                          * width / self.image_rect.width())),
            max(0.0, min(float(height), (position.y() - self.image_rect.y())
                           * height / self.image_rect.height())),
        )

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#171d27"))
        if self.image.isNull():
            painter.setPen(QColor("#96a2b4"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "等待相机画面\n连接成功后将在此显示实时预览")
            return
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        self.image_rect = QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)
        painter.drawImage(self.image_rect, self.image)
        painter.translate(self.image_rect.topLeft())
        painter.scale(scale, scale)
        painter.setPen(QPen(QColor("#57e6b5"), 2.5 / scale))
        painter.setBrush(QColor(87, 230, 181, 24))
        if self.polygon is not None:
            painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in self.polygon]))
        if self.drag_start is not None and self.drag_end is not None:
            painter.drawRect(QRectF(self.drag_start, self.drag_end).normalized())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton and self.selecting:
            self.cancel_selection()
        elif (self.selecting and event.button() == Qt.MouseButton.LeftButton
              and self.image_rect.contains(event.position())):
            self.drag_start = self._source_position(event.position())
            self.drag_end = self.drag_start
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self.selecting and self.drag_start is not None:
            self.drag_end = self._source_position(event.position())
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if not self.selecting or self.drag_start is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self.drag_end = self._source_position(event.position())
        rectangle = QRectF(self.drag_start, self.drag_end).normalized()
        x1, y1 = math.floor(rectangle.left()), math.floor(rectangle.top())
        x2, y2 = math.ceil(rectangle.right()), math.ceil(rectangle.bottom())
        if x2 - x1 < MIN_SELECTION_SIZE or y2 - y1 < MIN_SELECTION_SIZE:
            self.selection_rejected.emit("选区至少需要 32 × 32 像素，请重新拖选。")
            self.drag_start = self.drag_end = None
            self.update()
            return
        crop = self.frame[y1:y2, x1:x2].copy()
        self.cancel_selection()
        self.selected.emit(crop)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self.selecting:
            self.cancel_selection()
        else:
            super().keyPressEvent(event)
