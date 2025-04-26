# 딥러닝 모델의 기본 구성 요소

| 구성 요소 | 설명 | 예시 |
| --- | --- | --- |
| Linear(Dense) layer | 선형 변환 (행렬곱) | y = Wx + b |
| Activation function | 비선형성 추가 | ReLU, Sigmoid, Tanh, GELU 등 |
| Normalization | 데이터 분포를 정리 | BatchNorm, LayerNorm |
| Dropout | 과적합 방지용 랜덤 삭제 | Dropout |
| Residual connection | 정보 흐름 보강 | Transformer, ResNet |
| Attention mechanism | 특정 부분에 집중 | Self-Attention |

### 핵심 패턴을 요약

- Linear -> Activation -> Linear -> Activation -> ... 반복
+ Normalization/Dropout/Residual 추가
+ (필요하면 Attention 블록 삽입)

- 결국 **Linear(선형 변환)과 Activation(비선형 변환)의 조합이 뼈대**
- **이걸 어떻게 조립하느냐가 모델의 차별점**
- 벽돌(Linear + Activation)은 같지만, 집을 짓는 설계 방식이 다른 것
