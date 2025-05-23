# K-Means Clustering (K-평균 군집화)

<br>

## 핵심 개념

- 레이블이 없는 테이터를 유사한 특징끼리 묶는 비지도 학습 알고리즘
- K개의 중심점(centroid)를 기준으로 데이터를 나누고, 반복적으로 군집(cluster)을 개선
- 비유 : 학생들의 키와 체중을 기준으로 기준점을 만들고, 비슷한 친구들끼리 묶기

<br>

## 작동 원리 (반복 최적화 과정)

1. 원하는 군집 개수 K를 정함
2. 랜덤하게 K개의 중심점(centroid)을 초기화
3. 각 데이터를 가장 가까운 중심점에 할당
4. 각 군집의 중심점을 데이터 평균으로 업데이트
5. 중심점이 더 이상 변하지 않을 때까지 3~4번 반복

<br>

## 수학적 표현

- 목적 : 각 데이터와 중심점 간의 거리의 제곱합을 최소화

![K-NN_exp](K-NN_exp.png)

- Ci : i 번째 군집
- 𝜇i : i 번째 군집의 중심점
- ||x - 𝜇i||^2 : 거리의 제곱 (보통 유클리디안 거리 사용)

<br>

## 예시 코드

```python
from sklearn.cluster import KMeans
import numpy as np

# 예시 데이터 (공부시간, 수면시간)
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# K=2로 군집화
model = KMeans(n_clusters=2, random_state=42)
model.fit(X)

print("라벨:", model.labels_)
print("중심점:", model.cluster_centers_)
```

<br>

## K-Mean의 장단점

| 장점 | 단점 |
| --- | --- |
| 구현이 간단, 계산 속도가 빠르고 효율적 | K값(클러스터 수)를 미리 알아야 함 -> 도메인 지식이 없으면 최적 K 결정 어려움 |
| 대규모 데이터일수록 계산 기반의 성능이 좋아짐 | 초기 중심값에 민감해 초기값에 따라 결과가 달라질 수 있음(`로컬 최적해` 가능성) |
| 시각화 및 해석이 쉬움 | 비구형(비선형) 데이터에 한계 -> 둥근 형태가 아닌 클러스터는 잘 구분 못함 |
| 중심을 반복 갱신하는 방식으로 비교적 빨리 수렴함 | 스케일(크기)에 민감 -> 거리 기반으로 정규화가 필수 |
| 간단한 계산으로 메모리 자원 소모 적음 | 이상치에 취약 -> 평균 기반이므로 극단값에 영향을 많이 받음 |
| 거리 기반 | 유클리드 거리 기반으로 직관적 | 군집 간 밀도 차이 고려 불가 -> 밀도나 분포가 다르면 부정확한 군집화 가능 |

### 로컬 최적해(local optimum)

- 전체 중에 최고는 아니지만 주변에서는 최고인 답을 찾게 될 때

<br>

## 관련 개념

| 개념 | 설명 |
| --- | --- |
| 중심점 | 각 군집의 중심(평균값) |
| 군집 수 K | 사전에 정해야 하며, 잘못 설정하면 성능 저하 |
| 정규화 | 변수 간 단위가 다를 경우 필요 |
| `엘보우 기법` | 적절한 K값을 찾기 위한 시각적 방법 |
| `실루엣 계수` | 군집의 품질을 평가하는 지표 (0~1 사이) |

#### 엘보우 기법(Elbow Method)

- 적절한 K값(군집 수)를 고르기 위한 시각적 도구
- K가 커질수록 당연히 군집 내부의 오차(SSE, WCSS)가 줄어듦
- 하지만 어느 순간부터는 덜 줄어들고, 굳이 늘릴 필요 없는 시점이 나옴.
- 팔꿈치 꺾이는 지점처럼 생김 => 적절한 K값인 지점

#### 실루엣 계수

- 군집을 얼마나 잘 만들었는지 정량적으로 평가하는 방법
- 값은 0~1 사이이고, 1에 가까울수록 좋은 군집화
- 계산 방법 : silhouette = (b - a)/max(a, b)
    - a : 자기 클러스트 안에서 평균 거리(같은 군집에서 얼마나 떨어져 있는가)
    - b : 다른 클러스터들과의 거리 중 가장 작은 값(다른 군집과 얼마나 떨어져 있는가)
- 나랑 같은 그룹끼리는 가까워야하고,
- 나랑 다른 그룹은 멀어야 함.
