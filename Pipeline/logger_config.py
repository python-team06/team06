"""ML 파이프라인 공통 로깅 설정 모듈.

파이프라인의 학습, 평가, 저장, 보고서 생성 과정에서 발생하는 메시지를
터미널과 로그 파일에 동시에 기록하기 위해 사용한다.
"""

import logging

from src.config import LOG_DIR


def get_logger() -> logging.Logger:
    """ML 파이프라인 전용 로거를 생성해 반환한다.

    반환된 로거는 다음 두 위치에 같은 로그를 남긴다.

    1. 터미널 화면
    2. ``output/logs/ml_pipeline.log`` 파일

    같은 모듈이 여러 번 import되더라도 핸들러가 중복으로 추가되지 않도록
    기존 핸들러 존재 여부를 먼저 확인한다.
    """

    # 동일한 이름의 로거를 사용하면 프로젝트 전체에서 같은 설정을 공유할 수 있다.
    logger = logging.getLogger("ml_pipeline")

    # INFO 이상 수준의 로그를 기록한다.
    # DEBUG 메시지는 제외되고 INFO, WARNING, ERROR, CRITICAL은 기록된다.
    logger.setLevel(logging.INFO)

    # 이미 핸들러가 등록되어 있다면 그대로 반환한다.
    # 이 검사가 없으면 import될 때마다 동일한 로그가 여러 번 출력될 수 있다.
    if logger.handlers:
        return logger

    # 로그 파일은 config.py에서 생성한 output/logs 폴더 아래에 저장한다.
    log_file = LOG_DIR / "ml_pipeline.log"

    # 파일용 핸들러: 실행 이력을 UTF-8 형식으로 파일에 남긴다.
    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    # 터미널용 핸들러: 실행 상황을 콘솔에서 바로 확인할 수 있게 한다.
    stream_handler = logging.StreamHandler()

    # 모든 로그를 동일한 형식으로 출력한다.
    # 예: 2026-08-07 09:49:58,123 | INFO | 모델 평가 시작
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 로거에 파일 및 콘솔 핸들러를 연결한다.
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger