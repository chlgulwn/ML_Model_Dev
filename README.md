# ML_Model_Dev

## 사용한 데이터

- 데이터셋: `Breast Cancer Wisconsin Dataset`
- 출처: Scikit-learn 내장 데이터
- 특징 개수: 30개
- 샘플 수: 569개
- 목표: `양성(1)` / `악성(0)` 이진 분류


## 모델링 과정 요약

1. EDA 및 전처리
   - 이상치 제거 (IQR 이용)
   - 피처 선택 (SelectKBest, 20개)
   - Train/Test 분할
   - 스케일링 (StandardScaler)

2. 모델링
   - Logistic Regression
   - Support Vector Machine 
   - KNeighborsClassifier
   - Random Forest
   - Multi Layer Perceptron 

3. 성능 비교
   - 정확도, F1 점수, 5-Fold Cross Validation

4. 하이퍼파라미터 튜닝 (성능 높았던 두 모델 기반)
   - GridSearchCV 사용
   - Logistic Regression, MLP 최적화


## 5가지 모델 중 성능이 좋았던 Logistic Regression과 MLP를 가지고 Parameter Optimization 

| 모델                 | Test F1 score | Best Train F1 score |
|---------------------|---------------|----------------------|
| Logistic Regression |     0.960     |        0.979         |
| MLP Classifier      |     0.968     |        0.985         |

-> 최종적으로 해석력과 안정성을 고려하여 `Logistic Regression` 선택





## 사용 방법

```bash
# 가상환경 설치 후
pip install -r requirements.txt

# Jupyter에서 실행
jupyter notebook