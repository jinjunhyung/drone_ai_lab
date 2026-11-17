"""규칙 없이 "재고장 예측 모델 만들어 줘"라고만 했을 때 나오기 쉬운 스크립트 (감사 연습용 예시).

일부러 고치지 않았다. 블록 4에서 이 스크립트를 읽고 무엇이 문제인지 찾는다.
실행 (pdm_ai_lab 폴더):  python examples/v0_naive.py
                        python examples/v0_naive.py --this-month    # 이 모델로 이번 달 기록에 점수를 매겨 본다
"""
import sys

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("data/drone_train.csv", encoding="utf-8-sig")
df = df.fillna(df.mode().iloc[0])                                   # 결측은 가장 많이 나온 값으로
y = df["failure_within_30days"]
X = pd.get_dummies(df.drop(columns=["failure_within_30days", "maintenance_date"]))
X.columns = [f"f{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LGBMClassifier(n_estimators=300, random_state=42, verbose=-1).fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
print(f"정확도 {accuracy_score(y_test, prob >= 0.5):.3f} · ROC-AUC {roc_auc_score(y_test, prob):.3f}")
print("모델 학습을 마쳤습니다. 성능이 우수합니다.")

if "--this-month" in sys.argv:                                      # 이번 달(2026-01) 기록에 써 보기
    month = pd.read_csv("data/drone_this_month.csv", encoding="utf-8-sig")
    missing = [c for c in df.columns if c not in month.columns and c != "failure_within_30days"]
    print(f"\n이번 달 기록 {len(month)}건에 점수를 매깁니다 …")
    if missing:
        print(f"실패: 모델이 쓰는 컬럼이 이번 달 기록에 없습니다 → {missing}")
        print("이 값은 그 달이 끝나야 알 수 있습니다. 시험 점수는 나왔지만 실제로는 쓸 수 없는 모델입니다.")
