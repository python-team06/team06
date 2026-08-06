"""전체 모델 파이프라인 실행 파일."""


def main():
    # 같은 Pipeline 패키지 안의 pipeline.py를 불러온다.
    # pipeline.py의 최상위 코드가 순서대로 실행된다.
    from . import pipeline  # noqa: F401


if __name__ == "__main__":
    main()