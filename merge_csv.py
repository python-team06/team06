import pandas as pd


def main() -> None:
    # 1. CSV 파일 읽기
    results_df = pd.read_csv("results.csv")
    schema_df = pd.read_csv("schema.csv")

    # 2. 데이터 기본 정보 확인
    print("=" * 60)
    print("results.csv 앞 3행")
    print("=" * 60)
    print(results_df.head(3))

    print("\n" + "=" * 60)
    print("schema.csv 앞 3행")
    print("=" * 60)
    print(schema_df.head(3))

    print("\n" + "=" * 60)
    print("CSV 컬럼 정보")
    print("=" * 60)
    print("results.csv 컬럼 개수:", len(results_df.columns))
    print("schema.csv 컬럼 개수:", len(schema_df.columns))

    # 3. schema.csv에 qname 컬럼이 있는지 확인
    if "qname" not in schema_df.columns:
        raise ValueError("schema.csv에 'qname' 컬럼이 없습니다.")

    # 4. schema.csv의 qname 값 추출
    schema_qnames = set(
        schema_df["qname"]
        .dropna()
        .astype(str)
    )

    # 5. results.csv 컬럼 중 qname과 연결되는 컬럼만 선택
    # results.csv의 기존 컬럼 순서는 그대로 유지
    matched_columns = [
        column
        for column in results_df.columns
        if column in schema_qnames
    ]

    # 6. 연결되는 컬럼으로 새 DataFrame 생성
    matched_df = results_df[matched_columns].copy()

    # 7. CSV 파일 저장
    output_file = "matched_results.csv"

    matched_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("컬럼 연결 결과")
    print("=" * 60)
    print("연결되는 컬럼 개수:", len(matched_columns))
    print("연결되는 컬럼:", matched_columns)
    print("저장된 데이터 크기:", matched_df.shape)
    print(f"{output_file} 저장 완료")

    # 8. 생성된 CSV 파일 다시 읽기
    saved_df = pd.read_csv(output_file)

    print("\n" + "=" * 60)
    print("생성된 CSV 확인")
    print("=" * 60)
    print("생성된 CSV 컬럼 개수:", len(saved_df.columns))
    print("생성된 CSV 행 개수:", len(saved_df))
    print("생성된 CSV 크기:", saved_df.shape)
    print("생성된 CSV 컬럼:", saved_df.columns.tolist())
    print("\n생성된 CSV 앞 3행")
    print(saved_df.head(3))


if __name__ == "__main__":
    main()