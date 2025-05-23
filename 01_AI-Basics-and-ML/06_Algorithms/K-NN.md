# K-최근접 이웃 (K-Nearest Neighbors, K-NN)

<br>

## 핵심 개념

- 비슷한 데이터끼리는 비슷한 정답을 가질 것이다.
- 즉, 새로운 데이터가 주워졌을 때, 주변에 있는 데이터(k개)의 `레이블`을 보고 다수결로 결정

    - #### 레이블
        - 데이터에 딸려있는 정답 정보

        | 입력 데이터 (Input, X) | 레이블 (Label, 정답, Y)
        | --- | --- |
        | 어떤 사진 이미지 | 강아지 |
        | 어떤 사람의 키/몸무게 | 남자 or 여자 |
        | 시험 성적 80점 | 합격 or 불합격 |
        | 리뷰 내용 "정말 맛있어요!" | "긍정" |

<br>

## 작동 원리

- **가장 가까운 k개의 이웃을 보고, 그 중에서 가장 많은 레이블을 선택**

### 과정 정리

1. #### 학습(training)
    - 특정한 학습 없이,
    - 그냥 데이터를 저장
    - **lazy learning** 또는 instance-based learning 이라 부름

2. #### 예측(predicting):
    - 새로운 데이터 포인트 Xnew 가 들어옴
    - 기존 훈련 데이터들과의 거리(distance)를 계산
    - 가장 가짜운 k개의 데이터를 선택
    - 그 중 가장 많은 레이블을 가진 값으로 예측 결정

<br>

## 거리 계산 방식

| 거리 계산 방법 | 수식 | 설명 |
| --- | --- | --- |
| 유클리디안 거리(Euclidean, L2 Distance) | ((x1 - x2)^2 + (y1 - y2)^2)^(1/2) | 가장 기본적인 거리 계산
| 맨해튼 거리(Manhattan, L1 Distance) | \|x1 - x2\| + \|y1 - y2\| | 격자처럼 직각 방향으로만 움직인 거리 |
| 민코우스키 거리(Minkowski) | (∑ \|xᵢ - yᵢ\|^p)^(1/p) | p값에 따라 다른 거리로 일반화 (p=1이면 맨해튼, p=2이면 유클리디안) |

## 비유로 이해하기

> 시험을 처음 보는 학생이 친구에게 물어보는 상황

- 새로운 학생이 문제를 못품
- 근처에 있는 k명의 친구들에게 "너희는 답 뭐로 선택했어?" 라고 묻고,
- 가장 많은 친구가 말한 답을 따라서 정답으로 쓰는 것

<br>

## 예시

```python
from sklearn.neighbors import KNeighborsClassifier

# 데이터: (공부시간, 수면시간), 
# 레이블은 '합격/불합격'
X = [[8, 6], [7, 5], [3, 7], [2, 8], [1, 2]]
y = ['합격', '합격', '불합격', '불합격', '불합격']

# k=3인 모델 생성
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# 새로운 학생: 공부 4시간, 수면 6시간
print(model.predict([[4, 6]]))  # 출력: ['불합격']
```

<br>

## K-NN의 한계
| 문제 | 설명 |
| --- | --- |
| 계산량 많음 | 매번 거리 계산이 필요, 따라서 **데이터가 많을수록 느려짐** |
| 이상치 영향 | **주변에 이상한 데이터**가 있으면 예측이 틀릴 수 있음 | 
| 차원의 저주 | **고차원일수록 거리 계산이 애매**해짐 |

<br>

## 관련 개념

| 용어 | 설명 |
| --- | --- |
| k 값 | 너무 작으면 이상치에 민감, 너무 크면 경계가 흐려짐 |
| 거리 | 유클리리디안 외에도 다양한 거리 척도 사용 가능 | 
| 정규화 | Feature 간 단위가 다를 경우, 스케일링 필수 (예: `MinMaxScaler`) |

- ### MinMaxScaler
    - 정규화 : 머신러닝에서 입력 데이터의 값들이 서로 너무 다른 크기를 가진 경우, 거리 기반 알고리즘에서 한 쪽 Feature가 영향력을 독차지함
    - 그래서 각 특성(feature)의 값을 비슷한 범위로 바꿔주는 작업이 필요

    - 수식 : Xnorm = (x - xmin) / (xmax - xmin)
    - 각 데이터를 0과 1 사이의 값으로 스케일링
    - 예시 코드:
    ```python
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    # 예시 데이터 (2개 feature, 3개 샘플)
    X = np.array([[1, 200],
                [2, 300],
                [3, 400]])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print(X_scaled)
    # 출력 결과:
    # [[0.  0. ]
    #  [0.5 0.5]
    #  [1.  1. ]]
    ```