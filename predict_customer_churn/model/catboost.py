"""
CatBoost 모델 for Churn 예측 (이진 분류, ROC AUC).
main.py에서 MODEL_SELECT = "catboost" 로 선택해 사용.
"""
from catboost import CatBoostClassifier


def get_model(**kwargs):
    """
    main에서 사용할 모델 인스턴스 반환.
    반환 객체: .fit(X, y), .predict_proba(X) 지원.
    """
    default_params = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bylevel": 0.8,
        "random_state": 42,
        "verbose": 0,
        "eval_metric": "AUC",
    }
    default_params.update(kwargs)
    return CatBoostClassifier(**default_params)
