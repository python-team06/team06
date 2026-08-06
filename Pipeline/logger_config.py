import logging

from config import LOG_DIR


def get_logger() -> logging.Logger:
    """ML 파이프라인에서 사용할 로거를 생성한다."""

    logger = logging.getLogger("ml_pipeline")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    log_file = LOG_DIR / "ml_pipeline.log"

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger