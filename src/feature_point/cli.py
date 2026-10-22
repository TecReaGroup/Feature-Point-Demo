"""Desktop entry point."""

import os
import sys

from feature_point.config import PROJECT_ROOT, Settings
from feature_point.logging import configure_logging


def main() -> int:
    """Start the UI with repository-local caches and logging."""
    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "data/model"))
    logger = configure_logging()
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        from feature_point.recognition.window import RecognitionWindow
    except ModuleNotFoundError:
        logger.exception("Missing runtime dependency; install with make install")
        return 1
    except ImportError:
        logger.exception(
            "Runtime import failed; check DLL compatibility and dependency versions"
        )
        return 1
    application = QApplication(sys.argv)
    application.setApplicationName("Feature Point")
    application.setStyle("Fusion")
    try:
        settings = Settings.load()
        logger = configure_logging(settings.log_level)
        window = RecognitionWindow(settings)
    except Exception as exc:
        logger.exception("Application startup failed")
        QMessageBox.critical(None, "启动失败", str(exc))
        return 1
    window.show()
    logger.info("Application started")
    return application.exec()
