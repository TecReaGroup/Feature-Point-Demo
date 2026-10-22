"""Application logging and performance timing."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter_ns

PERF = 15
BEIJING_TIMEZONE = timezone(timedelta(hours=8))
LOG_DIRECTORY = Path("log")
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"

logging.addLevelName(PERF, "PERF")


class BeijingFormatter(logging.Formatter):
    """Format timestamps in UTC+08:00."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Render a timestamp with milliseconds for performance analysis."""
        timestamp = datetime.fromtimestamp(record.created, tz=BEIJING_TIMEZONE)
        if datefmt:
            return timestamp.strftime(datefmt)
        return f"{timestamp:%Y-%m-%d %H:%M:%S} +08:00"


class DailyFileHandler(logging.FileHandler):
    """Write records to files named by their Beijing calendar date."""

    def __init__(self) -> None:
        """Create the log directory and open today's log file."""
        LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        self.log_date = datetime.now(BEIJING_TIMEZONE).date()
        super().__init__(LOG_DIRECTORY / f"log_{self.log_date}.log", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        """Switch files when a record belongs to a different date."""
        record_date = datetime.fromtimestamp(record.created, tz=BEIJING_TIMEZONE).date()
        if record_date != self.log_date:
            self.close()
            self.log_date = record_date
            self.baseFilename = str((LOG_DIRECTORY / f"log_{self.log_date}.log").resolve())
            self.stream = self._open()
        super().emit(record)


@contextmanager
def perf_timer(logger: logging.Logger, operation: str) -> Iterator[None]:
    """Log elapsed milliseconds using a monotonic clock, including failed operations."""
    if not logger.isEnabledFor(PERF):
        yield
        return
    started_ns = perf_counter_ns()
    logger.log(PERF, "%s started", operation)
    try:
        yield
    finally:
        elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000
        logger.log(PERF, "%s finished elapsed_ms=%.3f", operation, elapsed_ms)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure console and daily file logging and return the application logger."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    formatter = BeijingFormatter(LOG_FORMAT)
    console_handler = logging.StreamHandler()
    file_handler = DailyFileHandler()
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
        existing_handler.close()
    for output_handler in (console_handler, file_handler):
        output_handler.setFormatter(formatter)
        root_logger.addHandler(output_handler)
    return logging.getLogger("feature_point")
