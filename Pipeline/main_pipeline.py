"""전체 모델 파이프라인을 시작하는 실행 진입점.

프로젝트 루트에서 다음 명령으로 실행한다.

    python -m pipeline.main_pipeline
"""


def main() -> None:
    """pipeline.py를 불러와 전체 파이프라인을 실행한다."""

    # 같은 pipeline 패키지 안의 pipeline.py를 import한다.
    # pipeline.py에 작성된 최상위 코드가 위에서 아래 순서로 실행되면서
    # 데이터 로딩, 모델 학습, 평가, 저장, 보고서 생성까지 진행된다.
    from . import pipeline  # noqa: F401

if __name__ == "__main__":
    main()