"""
LightGBM 모델 for Churn 예측 (이진 분류, ROC AUC).
main.py에서 MODEL_SELECT = "lgbm" 로 선택해 사용.
"""
from lightgbm import LGBMClassifier


def get_model(**kwargs):
    """
    main에서 사용할 모델 인스턴스 반환.
    반환 객체: .fit(X, y), .predict_proba(X) 지원.
    """
    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1,
        "metric": "auc",
    }
    default_params.update(kwargs)
    return LGBMClassifier(**default_params)
