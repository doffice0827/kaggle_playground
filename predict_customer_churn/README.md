# Predict Customer Churn

**Kaggle Playground Series - Season 6 Episode 3**

- **대회 링크**: [Predict Customer Churn](https://www.kaggle.com/competitions/playground-series-s6e3/overview)
- **태그**: Beginner, Tabular
- **평가 지표**: **ROC AUC Score**

---

## 대회 개요

통신사 고객의 **이탈(Churn)** 여부를 예측하는 이진 분류 대회입니다.  
고객 정보·서비스 이용 데이터를 바탕으로 이탈할 고객을 ROC AUC 기준으로 잘 구분하는 모델을 만드는 것이 목표입니다.

---

## 데이터

| 파일 | 설명 | 행 수(대략) |
|------|------|-------------|
| `data/train.csv` | 학습용 데이터 (타깃 포함) | ~594,196 |
| `data/test.csv` | 제출용 예측 대상 (타깃 없음) | ~254,657 |
| `data/sample_submission.csv` | 제출 형식 예시 | 동일 |

### 특징(Feature) 구성

| 컬럼 | 설명 | 예시 |
|------|------|------|
| `id` | 고객 식별자 | 0, 1, 2, ... |
| `gender` | 성별 | Male, Female |
| `SeniorCitizen` | 시니어 여부 | 0, 1 |
| `Partner` | 배우자 유무 | Yes, No |
| `Dependents` | 부양가족 유무 | Yes, No |
| `tenure` | 가입 기간(개월) | 1, 29, 58, ... |
| `PhoneService` | 전화 서비스 가입 | Yes, No |
| `MultipleLines` | 다중 회선 | Yes, No, No phone service |
| `InternetService` | 인터넷 서비스 종류 | DSL, Fiber optic, No |
| `OnlineSecurity` | 온라인 보안 | Yes, No, No internet service |
| `OnlineBackup` | 온라인 백업 | Yes, No, No internet service |
| `DeviceProtection` | 기기 보호 | Yes, No, No internet service |
| `TechSupport` | 기술 지원 | Yes, No, No internet service |
| `StreamingTV` | 스트리밍 TV | Yes, No, No internet service |
| `StreamingMovies` | 스트리밍 영화 | Yes, No, No internet service |
| `Contract` | 계약 기간 | Month-to-month, One year, Two year |
| `PaperlessBilling` | 무 paper 청구서 | Yes, No |
| `PaymentMethod` | 결제 수단 | Electronic check, Mailed check, ... |
| `MonthlyCharges` | 월 요금 | 20.2, 69.5, ... |
| `TotalCharges` | 누적 요금 | 20.2, 3778.2, ... |

### 타깃(Target)

- **`Churn`** (train에만 존재): 이탈 여부 — `Yes` / `No`  
- 제출 시에는 **0/1** 등 이진 값으로 예측해도 됩니다 (일반적으로 Yes=1, No=0).

---

## 제출 형식

`sample_submission.csv`와 동일한 형식:

```csv
id,Churn
594194,0
594195,0
...
```

- `id`: `test.csv`의 `id`와 동일
- `Churn`: 이탈 확률 또는 이진 레이블 (평가 지표가 ROC AUC이므로 **확률 예측** 제출이 일반적)

---

## 프로젝트 구조

```
predict_customer_churn/
├── README.md           # 이 파일
├── data_eda.ipynb      # EDA 노트북
└── data/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

---

## 참고

- **평가**: ROC AUC가 높을수록 좋음.
- **캐글 데이터**: 위 데이터는 워크스페이스 기준이며, 최신 데이터는 [대회 Data 탭](https://www.kaggle.com/competitions/playground-series-s6e3/data)에서 받을 수 있습니다.
