# Kaggle Playground

Kaggle Playground 시리즈 대회 참가용 레포지토리입니다.  
탐색적 데이터 분석(EDA), 피처 엔지니어링, 여러 모델 실험을 한 곳에서 관리합니다.

---

## 참가 대회 요약

| 대회 | 설명 | 평가 지표 | 태그 |
|------|------|-----------|------|
| **[Playground S6E3 – Predict Customer Churn](https://www.kaggle.com/competitions/playground-series-s6e3)** | 통신사 고객 이탈(Churn) 여부 이진 분류 | ROC AUC | Tabular, Beginner |

- **목표**: 고객 정보·서비스 이용 데이터로 이탈 고객을 ROC AUC 기준으로 잘 구분하는 모델 구축
- **데이터**: train ~59만 행, test ~25만 행, 범주형·수치형 혼합 + 합성 변수 활용

---

## 프로젝트 구조

```
kaggle_playground/
├── README.md                 # 이 파일 (전체 개요)
├── requirements.txt          # 공통 의존성 (pandas, sklearn, xgboost, catboost, lightgbm 등)
│
└── predict_customer_churn/   # Playground S6E3: 고객 이탈 예측
    ├── README.md             # 대회 상세 설명, 데이터 스키마, 제출 형식
    ├── main.py               # 학습·예측 실행 진입점 (DATA_SELECT, MODEL_SELECT 설정)
    ├── data_eda.ipynb        # EDA 및 합성 변수 검토
    ├── data/
    │   ├── train.csv
    │   ├── test.csv
    │   └── sample_submission.csv
    ├── model/
    │   ├── xgb.py            # XGBoost
    │   ├── catboost.py       # CatBoost
    │   └── lgbm.py           # LightGBM
    └── output/               # 제출 파일, run_log.csv
```

---

## 사용 방법

### 환경 설정

```bash
# 가상환경 권장
pip install -r requirements.txt
```

### 고객 이탈 예측 실행 (S6E3)

```bash
cd predict_customer_churn
python main.py
```

- **모델 선택**: `main.py` 상단에서 `MODEL_SELECT = "xgb"` | `"catboost"` | `"lgbm"` 중 선택
- **출력**: `output/submission_churn_{모델}_{타임스탬프}.csv` 생성, `run_log.csv`에 실행 이력 기록

---

## 기술 스택

- **언어**: Python 3.x
- **데이터**: pandas, numpy
- **전처리·평가**: scikit-learn
- **모델**: XGBoost, CatBoost, LightGBM
- **분석**: Jupyter, matplotlib, seaborn

---

## 기타

- 각 대회별 상세 내용(데이터 컬럼 설명, 제출 형식 등)은 해당 폴더의 `README.md`를 참고하세요.
- 캐글 최신 데이터는 각 대회의 [Data 탭](https://www.kaggle.com/competitions/playground-series-s6e3/data)에서 다운로드할 수 있습니다.
