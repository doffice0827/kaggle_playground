"""
학습·예측 실행 진입점.
상단에서 데이터와 모델만 바꿔서 실행하면 됨.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# ========== 여기만 바꿔서 사용 ==========
DATA_SELECT = "churn"   # 데이터: "churn" (추가 데이터는 아래 load_data에 등록)
MODEL_SELECT = "xgb"    # 모델: "xgb" (추가 모델은 get_model에 등록)
# ======================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)


def load_data(name: str):
    """
    데이터 이름에 따라 (X_train, y_train, X_test, test_ids) 반환.
    name == "churn" -> data/train.csv, data/test.csv 사용.
    """
    if name == "churn":
        return _load_churn()
    raise ValueError(f"Unknown data: {name}. Choose from: churn")


def _load_churn():
    """Churn 데이터 로드 및 전처리. train/test 동일 인코딩 적용."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # TotalCharges 숫자 변환 (공백 -> NaN)
    train["TotalCharges"] = pd.to_numeric(train["TotalCharges"], errors="coerce")
    test["TotalCharges"] = pd.to_numeric(test["TotalCharges"], errors="coerce")
    # 결측은 0 또는 중앙값으로 (tenure=0 구간과 맞춤)
    for df in (train, test):
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    target_col = "Churn"
    id_col = "id"
    drop_cols = [id_col, target_col]

    feature_cols = [c for c in train.columns if c not in drop_cols]
    # 수치형이 아닌 컬럼 = 범주형 (object, string, category 등 모두 포함)
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train[c])]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y_train = (train[target_col] == "Yes").astype(int).values
    test_ids = test[id_col].values

    # 범주형 인코딩 (train 기준 fit, train/test 동일 적용)
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
        X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))

    # float로 통일 (XGB 등에서 안정)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    return X_train, y_train, X_test, test_ids


def get_model(name: str):
    """모델 이름에 따라 fit(X,y), predict_proba(X) 지원하는 객체 반환."""
    if name == "xgb":
        from model.xgb import get_model as _get_xgb
        return _get_xgb()
    raise ValueError(f"Unknown model: {name}. Choose from: xgb")


def run():
    print(f"Data: {DATA_SELECT}, Model: {MODEL_SELECT}")
    print("Loading data...")
    X_train, y_train, X_test, test_ids = load_data(DATA_SELECT)
    print(f"Train {X_train.shape[0]} rows, {X_train.shape[1]} features. Test {X_test.shape[0]} rows.")

    print("Building model...")
    model = get_model(MODEL_SELECT)
    print("Training...")
    model.fit(X_train, y_train)

    print("Predicting...")
    # ROC AUC용 확률 (Churn=Yes 확률)
    proba = model.predict_proba(X_test)[:, 1]

    out_path = OUT_DIR / f"submission_{DATA_SELECT}_{MODEL_SELECT}.csv"
    sub = pd.DataFrame({"id": test_ids, "Churn": proba})
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()
