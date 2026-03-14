"""
학습·예측 실행 진입점.
상단에서 데이터와 모델만 바꿔서 실행하면 됨.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# ========== 여기만 바꿔서 사용 ==========
DATA_SELECT = "churn"   # 데이터: "churn" (추가 데이터는 아래 load_data에 등록)
MODEL_SELECT = "lgbm"    # 모델: "xgb" | "catboost" | "lgbm"
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


# EDA 상위 8개 합성변수 (MI 기준). train/test 동일 규칙 적용.
SYNTHETIC_FEATURES = [
    "is_monthly_contract",
    "avg_charge_per_month",
    "contract_x_payment",
    "tenure_x_monthly",
    "high_risk_segment",
    "log_totalcharges",
    "is_electronic_check",
    "is_fiber",
]


def _add_synthetic_features(df: pd.DataFrame) -> None:
    """상위 8개 합성변수 추가 (원본 컬럼 기준, in-place)."""
    df["is_monthly_contract"] = (df["Contract"] == "Month-to-month").astype(np.float32)
    df["avg_charge_per_month"] = (df["TotalCharges"] / np.maximum(df["tenure"], 1)).astype(np.float32)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(np.float32)
    df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(np.float32)
    df["contract_x_payment"] = (df["is_monthly_contract"] * df["is_electronic_check"]).astype(np.float32)
    df["tenure_x_monthly"] = (df["tenure"] * df["MonthlyCharges"]).astype(np.float32)
    df["high_risk_segment"] = (
        df["is_monthly_contract"] * df["is_electronic_check"] * df["is_fiber"]
    ).astype(np.float32)
    df["log_totalcharges"] = np.log1p(df["TotalCharges"].values).astype(np.float32)


def _load_churn():
    """Churn 데이터 로드 및 전처리. train/test 동일 인코딩 + 합성변수 적용."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # TotalCharges 숫자 변환 (공백 -> NaN)
    train["TotalCharges"] = pd.to_numeric(train["TotalCharges"], errors="coerce")
    test["TotalCharges"] = pd.to_numeric(test["TotalCharges"], errors="coerce")
    for df in (train, test):
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # 상위 8개 합성변수 추가 (EDA 결과 반영)
    _add_synthetic_features(train)
    _add_synthetic_features(test)

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
        from model.xgb import get_model as _get
        return _get()
    if name == "catboost":
        from model.catboost import get_model as _get
        return _get()
    if name == "lgbm":
        from model.lgbm import get_model as _get
        return _get()
    raise ValueError(f"Unknown model: {name}. Choose from: xgb, catboost, lgbm")


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

    ts = datetime.now().strftime("%y%m%d%H%M%S")
    out_path = OUT_DIR / f"submission_{DATA_SELECT}_{MODEL_SELECT}_{ts}.csv"
    sub = pd.DataFrame({"id": test_ids, "Churn": proba})
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # 실행 정보를 run_log.csv에 추가
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_row = pd.DataFrame([{
        "run_at": run_ts,
        "timestamp_id": ts,
        "data": DATA_SELECT,
        "model": MODEL_SELECT,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_features": X_train.shape[1],
        "synthetic_features": ",".join(SYNTHETIC_FEATURES),
        "synthetic_count": len(SYNTHETIC_FEATURES),
        "output_file": out_path.name,
    }])
    log_path = OUT_DIR / "run_log.csv"
    if log_path.exists():
        log_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_row.to_csv(log_path, mode="w", header=True, index=False)
    print(f"Run log appended: {log_path}")


if __name__ == "__main__":
    run()
