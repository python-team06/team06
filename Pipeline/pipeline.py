import numpy as np
import pandas as pd

from src.config import (
    AI_TEST_PARQUET,
    AI_TRAIN_PARQUET,
)

print("실행 중인 파일:", __file__)
print("학습 데이터:", AI_TRAIN_PARQUET)
print("테스트 데이터:", AI_TEST_PARQUET)

df = pd.read_parquet(AI_TRAIN_PARQUET)

# %%
# 데이터의 전체 구조를 빠르게 확인하기 위해 크기, 컬럼명, dtype, 샘플 행을 출력한다.
print("Data shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes.to_frame("dtype"))

print("\nData samples:")
print(df.head())

# %%
# 분류 문제이므로 target을 범주형 타입으로 바꿔서 이후 분석과 모델링에 일관되게 사용한다.
df["target"] = df["target"].astype("category")

# %%
import numpy as np

# 수치형 변수와 범주형 변수를 분리해 이후 EDA와 모델링 준비에 활용한다.
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric variables:", len(numeric_columns), numeric_columns)
print("Categorical variables:", len(categorical_columns), categorical_columns)

# %%
# 수치형 변수는 평균, 표준편차, 분위수 등을 보고 분포와 이상치를 확인한다.
print(df[numeric_columns].describe().T)

# 범주형 변수는 각 범주의 분포와 희소성을 확인해 모델 입력 전 점검한다.
print(df[categorical_columns].describe().T)

# %%
# 범주형 변수마다 등장 빈도와 비율을 확인해 희소한 범주나 치우침을 살펴본다.
cat_cols = df.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    print(f"\n[{col}]")
    dist = df[col].value_counts(dropna=False)
    ratio = df[col].value_counts(normalize=True, dropna=False) * 100
    result = pd.DataFrame({
        "count": dist,
        "ratio_%": ratio.round(2)
    })
    print(result)

# 범주가 너무 희소하면 모델이 불안정할 수 있으므로, 필요하면 기타 범주로 묶는 전략을 고려한다.

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from catboost import CatBoostClassifier

# train/test parquet를 각각 불러와 모델 학습과 최종 평가에 사용한다.
train_df = df.copy()
##test_df = pd.read_parquet("ai_test.parquet")
test_df = pd.read_parquet(AI_TEST_PARQUET)

# target은 범주형으로 유지하고, CatBoost 학습용 y는 숫자 코드로 변환한다.
train_df["target"] = train_df["target"].astype("category")
test_df["target"] = test_df["target"].astype("category")

# target을 제외한 나머지 컬럼을 입력 변수로 사용한다.
feature_columns = [col for col in train_df.columns if col != "target"]
X_train = train_df[feature_columns].copy()
y_train = train_df["target"].cat.codes
X_test = test_df[feature_columns].copy()
y_test = test_df["target"].cat.codes

# 숫자형과 그 외 컬럼을 구분해 CatBoost가 범주형으로 처리할 열을 따로 지정한다.
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [col for col in feature_columns if col not in numeric_features]
remaining_features = [col for col in feature_columns if col not in numeric_features + categorical_features]
if remaining_features:
    print('Uncovered features:', remaining_features)
    categorical_features.extend(remaining_features)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Covered feature count: {len(numeric_features) + len(categorical_features)}")
print(f"Target classes: {list(train_df['target'].cat.categories)}")

# 범주형 결측치는 문자열 sentinel로 채워 CatBoost가 그대로 받을 수 있게 한다.
def prepare_features(frame):
    prepared = frame.copy()
    for col in categorical_features:
        prepared[col] = prepared[col].astype('string').fillna('__MISSING__')
    return prepared

cat_feature_names = categorical_features.copy()

# CatBoost는 범주형 변수가 많은 데이터에서 강점이 있어, 별도 원-핫 인코딩 없이 그대로 학습한다.
model = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='Accuracy',
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=False,
    cat_features=cat_feature_names
)

pipeline = Pipeline([ # 전처리와 모델까지의 흐름을 한 번에 정리
    ('prepare', FunctionTransformer(prepare_features, validate=False)),
    ('model', model)
])

# %%
# 학습 데이터를 fit하고, test 데이터에서 예측값과 확률을 만든 뒤 성능을 확인한다.
pipeline.fit(X_train, y_train)

test_pred = pipeline.predict(X_test)
test_proba = pipeline.predict_proba(X_test)[:, 1] 

# accuracy와 confusion matrix, classification report로 분류 성능을 요약한다.
acc = accuracy_score(y_test, test_pred)
print(f"Test accuracy: {acc:.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nClassification report:")
print(classification_report(y_test, test_pred))

# 예측 결과를 실제 정답과 함께 확인할 수 있도록 표로 정리한다.
prediction_result = test_df[["target"]].copy()
prediction_result["predicted_target"] = test_pred
prediction_result["predicted_probability"] = test_proba
print(prediction_result.head())
#######################################
# 아래에서는 이미 학습된 pipeline을 이용해 저장·검증·그래프·HTML 보고서만 생성한다.
from .ml_pipeline import finalize_ml_pipeline

pipeline_result = finalize_ml_pipeline(
    model_pipeline=pipeline,
    X_test=X_test,
    y_test=y_test,
    test_df=test_df,
)

print("\n자동화 결과:")
print(pipeline_result)

