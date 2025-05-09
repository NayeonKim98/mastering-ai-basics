## 과적합(Overfitting)이란?

- 정의 : 모델이 **학습 데이터(training data)에 너무 지나치게 맞춰져서**, 새로운 데이터(test data)에서 성능이 떨어지는 현상 - noise or random fluctuations

- 비유 : 암기생 - 문제집의 정답만 달달 외우고 시험봤을 때, 살짝만 응용된 문제가 나와도 틀릴 가능성이 높음

### 과적합의 징후
| 지표 | 설명 |
|---|---|
| 훈련 정확도 매우 높음 | 학습 데이터에 엄청 잘 맞춤 |
| 검증 정확도 매우 낮음 | 그러나 새로운 데이터(검증/테스트셋)에선 잘 못 맞춤|
| 학습 그래프 | 훈련 오차는 계속 줄어드는데, 검증 오차는 **어느 순간부터 증가** |

### 과적합을 방지하는 9가지 방법
| 방법 | 설명 |
| ----- | ----- |
| 1. 더 많은 데이터 수집 | 데이터가 많으면 노이즈 < 패턴을 잘 학습 |
| 2. 정규화(Regularization) | 불필요한 복잡함을 **패널티**로 억제 - `L1/L2 정규화`|
| 3. `Dropout`(딥러닝) | 학습 중 일부 **뉴런을 랜덤하게 꺼서** 과잉적합 방지 |
| 4. 조기 종료(Early Stopping) | 검증 성능이 나빠질 조짐이 보이면 학습을 멈춤 |
| 5. 단순한 모델 사용 | 너무 복잡한 모델(고차 다항식 등)은 피함 |
| 6. 데이터 증강(Data Augmentation) | 이미지 등에서 회전, 반사 등을 이용해 데이터를 더 다양하게 사용 |
| 7. 교차 검증 | 데이터를 여러 부분으로 나누어 반복적으로 학습/검증하여 모델의 일반화 성능을 평가 |
| 8. `앙상블(Ensemble)` | 여러 모델의 예측을 결합하여 과적합을 줄이고 예측력을 높임 |
| 9. 특성 선택/차원 축소 | 불필요한 입력 특성을 제거하여 모델이 중요한 정보에만 집중할 수 있도록 함 |

### 정규화(Regularization)
- 모델이 너무 복잡해지지 않도록 벌점을 주는 방식
- 이 벌점은 모델이 너무 **치우치게 학습**하는 것을 방지
- 목표 : **단순하지만 잘 맞는** 모델을 만드는 것

    ### L1 정규화 vs L2 정규화
    | 항목 | L1 정규화 (Lasso) | L2 정규화 (Ridge) |
    |------|------------------|-------------------|
    | 수식 | Loss + λ * Σ&#124;w&#124; | Loss + λ * Σw² |
    | 벌점 | 가중치의 **절댓값**에 벌점 | 가중치의 **제곱**에 벌점 |
    | 효과 | **가중치가 0**이 되는 경우 많음 - 변수 선택 효과 | 가중치가 작아짐 - 변수가 남지만 영향이 줄어듬 |
    | 비유 | 안 쓰는 가구를 다 버리는 미니멀리스트 | 덜 중요한 부분은 작게 말하시는 선생님 |
    | 특징 | **Feature Selection** 효과 | 모든 특성을 **조금씩 살려** 사용 |

    ### Dropout
    - 정의 : 신경망 학습 시 **임의의 뉴런들을 꺼서** 학습하는 방식

    - 매 반복(epoch)마다 다른 구성의 네트워크를 학습하게 만들어 과적합 방지

    - 작동 방식 : 학습할 때, **각 뉴런을 일정 확률(예: 0.5)로 꺼버림**.
    
    - 효과 : 모델이 특정 뉴런에 너무 의존하지 않고 다양한 조합의 뉴런을 활용하도록 유도

    ### 앙상블(Ensemble)
    - 정의 : 여러 개의 모델을 결합해 하나의 강력한 모델을 만드는 기법
    - 단일 모델 하나보다 더 높은 정확도
    - 비유 : 집단지성

        #### 주요 앙상블 기법 3가지

        | 기법 | 개념 | 비유 | 효과 | 대표 알고리즘 |
        | --- | --- | --- | --- | --- |
        | Bagging | 여러 모델을 병렬로 훈련시키고 예측할 때 **여러 모델의 예측을 평균 or 투표** | 투표(서로 다른 전문가들의 의견 종합) | 데이터 샘플의 변동성이 큰 모델에 효과적 | `Random Forest` |
        | Boosting | **실수한 데이터에 집중**하여 실수를 줄여가며 순차적으로 학습 | 가르치는 선생님이 점점 틀린 문제 중심으로 설명 | 편향(bias)을 줄이는데 효과적 | `XGBoost`, `AdaBoost`, `LightGBM` |
        | Stacking | **여러 모델의 예측값을 다시 학습**한 결과를 메타모델이 최종 예측 | 전문가들 예측을 모아 종합적으로 분석하는 전문가를 더 두는 것 | 각 모델의 강점을 최대한 뽑아 쓰며, 서로 다른 유형의 모델도 가능(예: SVM + Tree + Neaural Net) | `Stacked Generalization` |


### 정리 문장

> 과적합(overfitting)은 모델이 훈련 데이터에 너무 맞춰져 새로운 데이터에 일반화하지 못하는 현상이며, 이를 해결하기 위해선 모델 단순화, 정규화, 조기 종료 등의 기법을 활용해야 한다.