# ML_Model

## 사용한 데이터

- 데이터셋: `Breast Cancer Wisconsin Dataset`
- 출처: Scikit-learn 내장 데이터
- 특징 개수: 30개
- 샘플 수: 569개
- 목표: `양성(1)` / `악성(0)` 이진 분류


## 과정 및 결과 요약

### 1. EDA 및 전처리
   - 이상치 제거 (IQR 이용)
   - 피처 선택 (SelectKBest, 20개)
   - Train/Test 분할
   - 스케일링 (StandardScaler)

### 2. 모델링
   - Logistic Regression
   - Support Vector Machine 
   - KNeighborsClassifier
   - Random Forest
   - Multi Layer Perceptron 

### 3. 성능 비교
   - 정확도, F1 점수, 5-Fold Cross Validation

### 4. 하이퍼파라미터 튜닝 (성능 높았던 두 모델 기반)
   - GridSearchCV 사용
   - Logistic Regression, MLP 최적화


### <5가지 모델 중 성능이 좋았던 Logistic Regression과 MLP를 가지고 Parameter Optimization한 결과> 

| 모델                 | Test F1 score | Best Train F1 score |
|---------------------|---------------|----------------------|
| Logistic Regression |     0.960     |        0.979         |
| MLP Classifier      |     0.968     |        0.985         |

-> 최종 모델로 더 높은 test F1 score를 기록한 `Multi Layer Perceptron(MLP)` 선택


### 5. 최종 테스트셋 결과 출력 
   - 안정적으로 성능이 좋아 선택한 'MLP'를 test dataset에 적용
   - 실제 레이블과 비교해보고, 예측의 정확도 등을 파악


--------------------------------------------------------------------------------------


# AutoML_Model

## 사용한 데이터

ML_Model과 동일한 `Breast Cancer Wisconsin Dataset`


## 과정 및 결과 요약

### 1. 전처리
- 데이터셋 로드 및 pandas DataFrame 변환
- 결측치 없음 확인
- Train/Test 분할 (`test_size=0.2`)
- 클래스 불균형 확인 (양성 285, 악성 170 → 불균형 심하지 않음)

### 2. AutoML 도구: `MLJAR-supervised`
- 모드: Perform (자동 탐색 + 최적화)
- 알고리즘: CatBoost, XGBoost, LightGBM
- 검증 전략: 5-Fold Cross Validation
- 기능: 앙상블 및 스태킹 포함 (`stack_models=True`, `train_ensemble=True`)

### 3. 평가 지표
- 주요 지표: F1 Score, Log Loss, Precision, Recall, Accuracy
- Assignment 1의 MLP 결과와 성능 비교


### <주요 결과>

|             Model                    |  Log Loss  | Test F1 Score  |
|--------------------------------------|------------|----------------|
| Best Single Model (CatBoost)         |  0.0836    |       -        |
| Best AutoML Model (Stacked Ensemble) | **0.0773** |   **0.966**    |
| Assignment 1 - MLP                   |     -      |   **0.968**    |

> test데이터셋의 f1스코어는 Assignment 1의 MLP가 살짝 더 높음.   
> AutoML은 다양한 알고리즘의 자동 실험과 앙상블을 통해 최적 모델을 효율적으로 도출함.  
> 특히, 이번 결과에서 **스태킹 앙상블 모델**이 가장 우수한 성능을 보여줌.



### 4. 최종 모델 및 테스트 결과

- 최종 선택된 모델: **Ensemble**  
  (CatBoost, XGBoost, LightGBM의 조합으로 구성된 앙상블)
- 테스트셋 F1 Score: **96.6%**
- 클래스별 Precision / Recall 균형 양호


### 5. 결과 파일 목록

- `AutoML_1/ldb_performance.png`: 성능 시각화 이미지
- `AutoML_1/leaderboard.csv`: 전체 모델 성능 비교표
- `AutoML_1/params.json`: 실험 설정 및 모델 구성 정보

--------------------------------------------------------------------------------------


## 사용 방법

```bash
# 가상환경 설치 후
pip install -r requirements.txt

# Jupyter에서 실행
jupyter notebook
