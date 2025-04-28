# 1. Activation Function 이란?

- 입력받은 값을 기반으로 출력 값을 결정하는 함수
- 즉, 들어온 값(input)을 똑똑하게 가공해서(output) 다음 층(layer)으로 넘겨주는 역할

### 왜 필요할까?
- **비선형성** 부여
    - 세상은 직선처럼 단순하지 않음. 만약 선형 함수만 있다면, 아무리 층을 많이 쌓아도 결국은 하나의 선형 함수로 뭉개질 것임. 
    - 비선형성 활성 함수로 모델이 복잡한 패턴 학습 가능

# 2. 주요 Activation Functions

| 함수 이름 | 수식 | 특징 | 비유 |
| --- | --- | --- | --- |
| Sigmoid | σ(x) = 1/(1+e^(-x)) | 0~1 사이 출력, `gradient vanishing` 문제 | 출구가 좁은 터널 |
| Tanh | tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)) | -1~1 출력, `Zero-centered` | 양방향 미끄럼틀 |
| ReLU | f(x) = max(0, x) | 간단, 빠름, `죽은 뉴런` 문제 | 수도꼭지(잠기거나 쭉 뿜거나) |
| Leaky ReLU | f(x) = max(0.01x, x) | 음수 입력도 살짝 반영 | 살짝 새는 수도꼭지 |
| ELU | f(x) = x (x > 0), α(e^x −1) (x =< 0) | ReLU 개선형, smooth output | 고무줄처럼 부드러운 문 |
| Softmax | Softmax(xi) = e^xi / ∑je^xj | 다중 클래스 분류용, 확률로 해석 가능 | 투표 결과 퍼센트로 나누기 |

# 3. 대표 Activation Function

## 3.1 Sigmoid 

- 특징 : 출력값이 항상 (0, 1) 사이
- 사용 예시 : 이진 분류 문제의 마지막 층에서 주로 사용
- 단점 : 입력값이 너무 크거나 작으면 gradient가 0에 가까워져서 학습이 잘 안됨. 이게 gradient vanishing

## 3.2 Tanh

- 특징 : 출력값이 (-1, 1) 사이. 그래서 중심이 0이라 학습이 Sigmoid보다 빠름
- 단점 : 여전히 gradient vanishing 문제
- Tanh = 강화된 Sigmoid

## 3.3 ReLU (Rectified Linear Unit)

- 특징 : 양수면 그대로, 음수면 0
- 단점 : 음수 입력이 계속되면 뉴런이 죽어버림 -> dead neuron 문제

## 3.4 Leaky ReLU

- 특징 : ReLU의 단점 살짝 보완. 음수 구간 0이 아니고 0.01배 정도 남겨 죽은 뉴런 문제 완화

## 3.5 ELU (Exponential Linear Unit)

- 특징 : Leaky ReLU보다 더 부드럽게, 자연스럽게 이어짐. 양수는 직선, 음수는 지수 함수 형태
- 장점 : Smooth transition 덕분에 학습이 잘되는 경우 많음

## 3.6 softmax 

- 특징 : 다수의 클래스에서 각 클래스가 정답일 확률을 구해주는 함수
- 출력값들의 총합이 1
- 사용 예시 : 다중 분류 문제 마지막층에 항상 등장

# 4. Activation Function 선택 가이드

| 상황 | 추천 함수 |
| --- | --- |
| 빠른 연산 필요 | ReLU |
| 음수와 양수 둘 다 고려 | Tanh |
| 부드럽게 넘어가기 | ELU |
| 이진 분류 최종 출력 | Sigmoid |
| 다중 분류 최종 출력 | Softmax |

