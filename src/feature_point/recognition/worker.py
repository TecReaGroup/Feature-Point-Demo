"""Bounded camera and inference workers with latest-frame mailboxes."""

import importlib.util
import logging
import threading
import time

from PySide6.QtCore import QThread, Signal

from feature_point.config import PROJECT_ROOT, CameraSettings, RecognitionSettings

LOGGER = logging.getLogger(__name__)
CAMERA_TIMEOUT_SECONDS = 5.0


class CameraWorker(QThread):
    """Adapt the repository's device/UsbCamera.py driver without blocking Qt."""

    failed = Signal(str)
    ready = Signal(str)

    def __init__(self, settings: CameraSettings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.lock = threading.Lock()
        self.latest = None

    def take_frame(self):
        """Consume the newest frame, dropping overwritten captures."""
        with self.lock:
            frame, self.latest = self.latest, None
        return frame

    def run(self) -> None:
        """Open, sample and release the camera entirely outside the GUI thread."""
        camera = None
        try:
            specification = importlib.util.spec_from_file_location(
                "usb_camera", PROJECT_ROOT / "device/UsbCamera.py"
            )
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            camera = module.UsbCamera({
                "deviceId": self.settings.index, "backend": self.settings.backend,
                "colorImageSizeX": self.settings.width,
                "colorImageSizeY": self.settings.height, "fps": self.settings.fps,
            })
            camera.open()
            self.ready.emit(f"CAM {self.settings.index:02d} · {camera.backend}")
            last_capture = time.perf_counter()
            while not self.isInterruptionRequested():
                captured_at, frame = camera.getFrame(timeout=0.1)
                if frame is not None:
                    last_capture = captured_at
                    with self.lock:
                        self.latest = (captured_at, frame)
                elif time.perf_counter() - last_capture > CAMERA_TIMEOUT_SECONDS:
                    raise RuntimeError("相机连续 5 秒未返回图像，请检查连接后重连。")
        except Exception as exc:
            LOGGER.exception("Camera failed")
            self.failed.emit(str(exc))
        finally:
            if camera is not None:
                camera.stopThread()
            LOGGER.info("Camera stopped")


class RecognitionWorker(QThread):
    """Keep one pending frame; never queue an unbounded inference backlog."""

    ready = Signal(str)
    failed = Signal(str)
    reference_ready = Signal(int, int)
    reference_rejected = Signal(int, str)

    def __init__(self, settings: RecognitionSettings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.condition = threading.Condition()
        self.revision = 0
        self.reference = None
        self.pending_frame = None
        self.latest_detection = None

    def set_reference(self, crop) -> int:
        """Invalidate old output immediately when selecting or clearing a reference."""
        with self.condition:
            self.revision += 1
            self.reference = crop
            self.pending_frame = None
            self.latest_detection = None
            self.condition.notify_all()
            return self.revision

    def submit(self, captured_at: float, frame) -> None:
        """Replace pending input with the newest camera capture."""
        with self.condition:
            self.pending_frame = (captured_at, frame)
            self.condition.notify_all()

    def take_detection(self):
        """Consume the latest completed localization."""
        with self.condition:
            detection, self.latest_detection = self.latest_detection, None
            return detection

    def stop(self) -> None:
        """Wake an idle worker so shutdown can finish cooperatively."""
        self.requestInterruption()
        with self.condition:
            self.condition.notify_all()

    def run(self) -> None:
        """Initialize GPU models and process frames without touching any widget."""
        try:
            from feature_point.recognition.matcher import ObjectMatcher

            matcher = ObjectMatcher(self.settings)
            self.ready.emit(matcher.device_name)
            applied_revision = -1
            reference_valid = False
            while not self.isInterruptionRequested():
                with self.condition:
                    self.condition.wait_for(lambda: self.isInterruptionRequested()
                                            or self.revision != applied_revision
                                            or self.pending_frame is not None)
                    if self.isInterruptionRequested():
                        break
                    revision, reference = self.revision, self.reference
                    pending, self.pending_frame = self.pending_frame, None
                if revision != applied_revision:
                    applied_revision = revision
                    reference_valid = False
                    if reference is not None:
                        try:
                            count = matcher.set_reference(reference)
                            reference_valid = True
                            self.reference_ready.emit(revision, count)
                        except ValueError as exc:
                            LOGGER.warning("Reference rejected: %s", exc)
                            self.reference_rejected.emit(revision, str(exc))
                if not reference_valid or pending is None:
                    continue
                captured_at, frame = pending
                started = time.perf_counter()
                localization = matcher.locate(frame)
                elapsed = time.perf_counter() - started
                with self.condition:
                    if revision == self.revision:
                        self.latest_detection = (revision, captured_at, frame, localization, elapsed)
        except Exception as exc:
            LOGGER.exception("Recognition worker failed")
            self.failed.emit(str(exc))
        finally:
            LOGGER.info("Recognition stopped")
