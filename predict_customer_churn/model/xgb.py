"""
XGBoost 모델 for Churn 예측 (이진 분류, ROC AUC).
main.py에서 MODEL = "xgb" 로 선택해 사용.
"""
import numpy as np
from xgboost import XGBClassifier


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
        "eval_metric": "auc",
        "use_label_encoder": False,
    }
    default_params.update(kwargs)
    return XGBClassifier(**default_params)
