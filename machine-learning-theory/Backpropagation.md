# Backpropagation (역전파)의 정의

- 신경망이 학습할 때 오차를 계산하고, 이 오차를 거꾸로 전파해서 각 가중치를 얼마나 조정해야할지 알려주는 방법
- 실수를 찾고, 어디를 얼마나 고쳐야하는지 알려주는 메커니즘

# 왜 필요한가?

- 신경망은 예측을 하지만, 처음에는 엄청 틀림
- 그 틀린 정도를 오차라고 하고, 이 오차를 줄이려면 가중치를 조정해야함
- 그런데 가중치가 엄청 많으면(수백, 수천개)
- 어떤 가중치를 얼마나 바꿔야하는가? 이걸 계산하는게 역전파!

# 작동 원리

## (1) 순전파 (Forward)

- 입력을 받아
- 각 층을 지나며 계산하고
- 최종 출력을 만듦
- input -> hidden layer -> output

## (2) 오차 계산 (Error Computation)

- 출력과 정답과의 차이로 오차 계산
    - **오차 함수** 사용 (Kean Squared Error, Cross Entropy)

## (3) 역전파 (Backward)

- 오차를 출력층에서 입력층 방향으로 거꾸로 전파하면서
- 각각의 가중치가 오차에 얼마나 기여했는지 계산
- 수학적으로는, 미분 이용(얼마나 민감하게 변화하는지 측정)

## (4) 가중치 업데이트

- 계산한 기울기를 가지고
- 기울기를 살짝 조정한다 (gradient descent 방식)
- 가중치 조정 공식 : new weight = old weight - learning rate x gradient
    - learning rate = 얼마나 크게 수정할지 정하는 하이퍼파라미터

### 핵심 수학 공식

- 오치(loss)를 가중치(weight)로 미분해서 이 미분값(기울기)을 가지고 weight를 조정

# Backpropagation 과정

```scss
[Input Layer] → [Hidden Layer] → [Output Layer]
       ↓               ↓               ↓
     Forward         Forward         Compute Output
      Pass            Pass           and Loss(Error)
  
       ←    ←   ←   ←    ←   ←   ←   ←  
              Backpropagate Error
        Calculate Gradients(미분)
        Update Weights
```
