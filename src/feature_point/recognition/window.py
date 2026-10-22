"""Desktop workspace for single-reference object localization."""

import time

from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from feature_point.config import Settings
from feature_point.recognition.canvas import VideoCanvas, frame_image
from feature_point.recognition.worker import CameraWorker, RecognitionWorker

STALE_CAPTURE_SECONDS = 1.0
ALGORITHM_LABELS = {
    "aliked_lightglue": "ALIKED + LightGlue",
    "efficientloftr": "EfficientLoFTR",
}
STYLE = """
QMainWindow, QWidget#workspace { background: #f5f6f8; color: #202a38; }
QWidget { font-family: 'Segoe UI', 'Microsoft YaHei UI'; font-size: 13px; color: #202a38; }
QFrame#card { background: #ffffff; border: 1px solid #e3e7ed; border-radius: 12px; }
QLabel { background: transparent; border: none; }
QLabel#title { font-size: 23px; font-weight: 700; color: #182334; }
QLabel#muted { color: #778291; font-size: 12px; }
QLabel#section { font-size: 14px; font-weight: 600; color: #263244; }
QLabel#badge { color: #536277; background: #e9edf3; border-radius: 6px; padding: 6px 10px; font-size: 11px; }
QLabel#reference { background: #f1f3f6; border: 1px dashed #d4dbe5; border-radius: 8px; color: #8b96a5; }
QLabel#status { font-size: 19px; font-weight: 600; color: #263244; }
QLabel#metrics { background: #f5f7fa; border-radius: 8px; padding: 14px; color: #536277; }
QLabel#step { color: #526886; font-size: 11px; font-weight: 600; }
QPushButton { background: #ffffff; border: 1px solid #dce2ea; border-radius: 7px; padding: 9px 14px; font-weight: 500; }
QPushButton:hover { background: #f0f4fa; border-color: #b7c5d8; }
QPushButton:pressed { background: #e4ebf5; }
QPushButton:focus { border: 1px solid #3975dc; }
QPushButton:disabled { color: #a5aeba; background: #f5f6f8; border-color: #e8ebf0; }
QPushButton#primary { background: #326cdb; color: white; border: 1px solid #326cdb; font-weight: 600; }
QPushButton#primary:hover { background: #255bc0; }
QPushButton#primary:pressed { background: #204fa6; }
QPushButton#primary:disabled { background: #e3eaf6; color: #98abc9; border-color: #e3eaf6; }
QScrollArea#sidebar, QWidget#sidebarContent { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }
QScrollBar::handle:vertical { background: #cbd3df; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""


class RecognitionWindow(QMainWindow):
    """Coordinate camera preview, reference selection and versioned detections."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.setWindowTitle(f"Feature Point · {ALGORITHM_LABELS[settings.recognition.algorithm]}")
        self.resize(1320, 820)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(STYLE)
        self.closing = False
        self.model_ready = False
        self.reference_active = False
        self.reference_crop = None
        self.revision = 0
        self.camera_available = False
        self.last_capture = 0.0
        self.last_detection = 0.0
        self.previous_capture = None
        self.camera = None
        self._build_workspace()
        self.recognition = RecognitionWorker(settings.recognition, self)
        self.recognition.ready.connect(self._model_ready)
        self.recognition.failed.connect(self._model_failed)
        self.recognition.reference_ready.connect(self._reference_ready)
        self.recognition.reference_extracted.connect(self._reference_extracted)
        self.recognition.reference_rejected.connect(self._reference_rejected)
        self.recognition.start()
        self._start_camera()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(16)

    def _build_workspace(self) -> None:
        root = QWidget(objectName="workspace")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 16)
        layout.setSpacing(24)
        heading = QHBoxLayout()
        branding = QVBoxLayout()
        branding.setSpacing(4)
        branding.addWidget(QLabel("Feature Point", objectName="title"))
        branding.addWidget(QLabel("实时视觉定位工作台", objectName="muted"))
        heading.addLayout(branding)
        heading.addStretch()
        heading.addWidget(QLabel(ALGORITHM_LABELS[self.settings.recognition.algorithm], objectName="badge"))
        layout.addLayout(heading)
        content = QHBoxLayout()
        content.setSpacing(20)
        preview_card = QFrame(objectName="card")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_layout.setSpacing(14)
        preview_heading = QHBoxLayout()
        preview_heading.addWidget(QLabel("实时画面", objectName="section"))
        preview_heading.addStretch()
        self.capture_rate = QLabel("— FPS", objectName="muted")
        preview_heading.addWidget(self.capture_rate)
        preview_layout.addLayout(preview_heading)
        self.canvas = VideoCanvas()
        self.canvas.selected.connect(self._select_reference)
        self.canvas.selection_changed.connect(self._selection_changed)
        self.canvas.selection_rejected.connect(self._show_message)
        preview_layout.addWidget(self.canvas, 1)
        toolbar = QHBoxLayout()
        self.select_button = QPushButton("框选物体", objectName="primary")
        self.select_button.setEnabled(False)
        self.select_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.select_button.clicked.connect(self.canvas.begin_selection)
        toolbar.addWidget(self.select_button)
        toolbar.addStretch()
        self.reconnect = QPushButton("重连相机")
        self.reconnect.setEnabled(False)
        self.reconnect.clicked.connect(self._start_camera)
        toolbar.addWidget(self.reconnect)
        preview_layout.addLayout(toolbar)
        self.instruction = QLabel("点击「框选物体」，在画面上拖动选择目标", objectName="muted")
        self.instruction.setWordWrap(True)
        preview_layout.addWidget(self.instruction)
        content.addWidget(preview_card, 1)
        sidebar_scroll = QScrollArea(objectName="sidebar")
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(304)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_content = QWidget(objectName="sidebarContent")
        sidebar_scroll.setWidget(sidebar_content)
        sidebar = QVBoxLayout(sidebar_content)
        sidebar.setContentsMargins(0, 0, 6, 0)
        sidebar.setSpacing(16)
        reference_card = QFrame(objectName="card")
        reference_layout = QVBoxLayout(reference_card)
        reference_layout.setContentsMargins(16, 16, 16, 16)
        reference_layout.setSpacing(12)
        reference_layout.addWidget(QLabel("01  /  REFERENCE", objectName="step"))
        reference_heading = QHBoxLayout()
        reference_heading.addWidget(QLabel("参考物体", objectName="section"))
        reference_heading.addStretch()
        self.clear_button = QPushButton("清除")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self._clear_reference)
        reference_heading.addWidget(self.clear_button)
        reference_layout.addLayout(reference_heading)
        self.reference_image = QLabel("尚未选择物体\n\n在左侧画面中框选一个目标", objectName="reference")
        self.reference_image.setFixedHeight(176)
        self.reference_image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.reference_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reference_layout.addWidget(self.reference_image)
        self.reference_caption = QLabel("参考图固定保留，不随相机画面更新", objectName="muted")
        self.reference_caption.setWordWrap(True)
        reference_layout.addWidget(self.reference_caption)
        sidebar.addWidget(reference_card)
        status_card = QFrame(objectName="card")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(16, 16, 16, 16)
        status_layout.setSpacing(14)
        status_layout.addWidget(QLabel("02  /  LOCALIZATION", objectName="step"))
        self.status = QLabel("正在加载模型…", objectName="status")
        self.status.setWordWrap(True)
        status_layout.addWidget(self.status)
        self.metrics = QLabel("匹配 —   /   内点 —\n推理 — ms", objectName="metrics")
        self.metrics.setMinimumHeight(78)
        status_layout.addWidget(self.metrics)
        self.device_label = QLabel("初始化计算设备…", objectName="muted")
        self.device_label.setWordWrap(True)
        status_layout.addWidget(self.device_label)
        sidebar.addWidget(status_card)
        hint = QLabel("选择纹理清晰的物体，定位更稳定。\n\n框选时画面冻结，Esc / 右键取消。\n目标离开画面后将自动持续搜索。", objectName="muted")
        hint.setWordWrap(True)
        hint.setContentsMargins(4, 4, 4, 4)
        sidebar.addWidget(hint)
        sidebar.addStretch()
        content.addWidget(sidebar_scroll)
        layout.addLayout(content, 1)
        footer = QHBoxLayout()
        footer.setSpacing(24)
        self.camera_label = QLabel("正在连接相机…", objectName="muted")
        self.camera_label.setMaximumWidth(340)
        self.camera_label.setWordWrap(True)
        footer.addWidget(self.camera_label)
        self.message = QLabel("首次启动将下载模型权重，请保持网络连接。", objectName="muted")
        self.message.setWordWrap(True)
        self.message.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.message.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        footer.addWidget(self.message, 1)
        layout.addLayout(footer)

    def _start_camera(self) -> None:
        if self.closing or (self.camera is not None and self.camera.isRunning()):
            return
        if self.camera is not None:
            self.camera.deleteLater()
        self.camera_available = False
        self.previous_capture = None
        self.reconnect.setEnabled(False)
        self.camera_label.setText("正在连接相机…")
        self.camera = CameraWorker(self.settings.camera, self)
        self.camera.ready.connect(self.camera_label.setText)
        self.camera.failed.connect(self._camera_failed)
        self.camera.finished.connect(self._camera_stopped)
        self.camera.start()

    def _camera_stopped(self) -> None:
        self.camera_available = False
        self.select_button.setEnabled(False)
        self.reconnect.setEnabled(not self.closing)

    def _camera_failed(self, message: str) -> None:
        self.camera_available = False
        self.canvas.cancel_selection()
        self.canvas.polygon = None
        self.canvas.update()
        self.camera_label.setText("相机断开")
        self.status.setText("等待相机重连")
        self._show_message(message)

    def _model_ready(self, device: str) -> None:
        self.model_ready = True
        self.device_label.setText(device)
        self.status.setText("请选择参考物体")
        self._show_message("模型已就绪。框选参考物体后开始实时搜索。")

    def _model_failed(self, message: str) -> None:
        self.model_ready = False
        self.reference_active = False
        self.canvas.cancel_selection()
        self.canvas.polygon = None
        self.canvas.update()
        self.status.setText("模型加载或推理失败")
        self.device_label.setText("请检查配置及日志，修复后重新启动")
        self._show_message(message)

    def _select_reference(self, selection) -> None:
        frame, (x1, y1, x2, y2) = selection
        crop = frame[y1:y2, x1:x2].copy()
        self.reference_crop = crop
        self.revision = self.recognition.set_reference(selection)
        self.reference_active = False
        self.reference_image.setPixmap(QPixmap.fromImage(frame_image(crop)).scaled(
            self.reference_image.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        self.reference_caption.setText(f"{crop.shape[1]} × {crop.shape[0]} px · 正在建立参考")
        self.clear_button.setEnabled(True)
        self.metrics.setText("匹配 —   /   内点 —\n推理 — ms")
        self.status.setText("正在建立参考…")

    def _reference_extracted(self, revision: int, points) -> None:
        """Overlay actual extracted points in crop-local preview coordinates."""
        if revision != self.revision:
            return
        pixmap = QPixmap.fromImage(frame_image(self.reference_crop)).scaled(
            self.reference_image.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#10251e"), 1))
        painter.setBrush(QColor("#57e6b5"))
        height, width = self.reference_crop.shape[:2]
        for x, y in points:
            painter.drawEllipse(QPointF(float(x) * pixmap.width() / width,
                                       float(y) * pixmap.height() / height), 2.5, 2.5)
        painter.end()
        self.reference_image.setPixmap(pixmap)
        if not self.reference_active:
            self.reference_caption.setText("各方向有效特征不足 · 请扩大选区")

    def _reference_ready(self, revision: int, count: int) -> None:
        if revision != self.revision:
            return
        self.reference_active = True
        self.reference_caption.setText(
            f"多角度参考已锁定 · 合并显示 {count} 个特征位置" if count
            else "参考已锁定 · 等待图像对匹配，绿色点将随匹配更新"
        )
        self.status.setText("正在搜索物体…")
        self._show_message("绿色轮廓表示通过几何验证的目标；可随时重新框选。")

    def _reference_rejected(self, revision: int, message: str) -> None:
        if revision == self.revision:
            self.reference_active = False
            self.reference_caption.setText("参考特征不足 · 请重新框选")
            self.status.setText("需要新的参考物体")
            self._show_message(message)

    def _clear_reference(self) -> None:
        self.canvas.cancel_selection()
        self.revision = self.recognition.set_reference(None)
        self.reference_active = False
        self.reference_image.clear()
        self.reference_crop = None
        self.reference_image.setText("尚未选择物体\n\n在左侧画面中框选一个目标")
        self.reference_caption.setText("参考图固定保留，不随相机画面更新")
        self.clear_button.setEnabled(False)
        self.canvas.polygon = None
        self.canvas.update()
        self.status.setText("请选择参考物体")
        self.metrics.setText("匹配 —   /   内点 —\n推理 — ms")

    def _selection_changed(self, selecting: bool) -> None:
        self.select_button.setEnabled(not selecting and self.model_ready and self.camera_available)
        self.instruction.setText("画面已冻结 · 拖动框选物体 · Esc / 右键取消" if selecting
                                 else "点击「框选物体」，在画面上拖动选择目标")

    def _show_message(self, message: str) -> None:
        self.message.setText(message)

    def _refresh(self) -> None:
        now = time.perf_counter()
        capture = self.camera.take_frame()
        if capture is not None:
            captured_at, frame = capture
            self.last_capture = captured_at
            self.camera_available = True
            if self.previous_capture is not None:
                interval = captured_at - self.previous_capture
                if interval > 0:
                    self.capture_rate.setText(f"{1 / interval:.0f} FPS · {frame.shape[1]} × {frame.shape[0]}")
            self.previous_capture = captured_at
            if self.reference_active and not self.canvas.selecting:
                self.recognition.submit(captured_at, frame)
            if not self.reference_active or now - self.last_detection > STALE_CAPTURE_SECONDS:
                self.canvas.set_frame(frame)
        detection = self.recognition.take_detection()
        if detection is not None and not self.canvas.selecting and self.camera_available:
            revision, captured_at, frame, localization, elapsed = detection
            if revision == self.revision:
                self.last_detection = now
                self.canvas.set_frame(frame, localization.polygon, localization.confidence)
                if localization.reference_points is not None:
                    self._reference_extracted(revision, localization.reference_points)
                    self.reference_caption.setText(
                        f"半稠密匹配 · 当前选区 {len(localization.reference_points)} 对对应点"
                    )
                self.status.setText("已找到目标" if localization.polygon is not None else localization.reason)
                self.metrics.setText(f"匹配 {localization.matches}   /   内点 {localization.inliers}\n"
                                      f"推理 {elapsed * 1000:.0f} ms · {1 / max(elapsed, 0.001):.1f} FPS\n"
                                      f"结果画面延迟 {max(0, now - captured_at):.1f} s")
        if (self.reference_active and self.camera_available
                and now - self.last_detection > STALE_CAPTURE_SECONDS):
            self.canvas.polygon = None
            self.canvas.update()
            self.status.setText("等待新定位结果…")
        if self.camera_available and now - self.last_capture > STALE_CAPTURE_SECONDS:
            self.camera_available = False
            self.canvas.polygon = None
            self.canvas.update()
            self.status.setText("相机画面中断…")
        self.select_button.setEnabled(self.model_ready and self.camera_available and not self.canvas.selecting)

    def closeEvent(self, event) -> None:
        """Wait asynchronously for workers; never destroy a running QThread."""
        if not self.closing:
            self.closing = True
            self.timer.stop()
            self.centralWidget().setEnabled(False)
            self.message.setText("正在关闭相机与推理线程…")
            self.camera.requestInterruption()
            self.recognition.stop()
        if self.camera.isRunning() or self.recognition.isRunning():
            event.ignore()
            QTimer.singleShot(100, self.close)
        else:
            event.accept()
