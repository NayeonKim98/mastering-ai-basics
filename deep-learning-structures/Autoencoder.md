# Autoencoder (오토인코더)란?

<br>

## 1. 정의

- Autoencoder는 입력 데이터를 압축(encode)하고 다시 복원(decode)하는 신경망(network) 구조

    - 즉 "데이터를 요약했다 다시 되살리는 것"

- Auto(자동) + Encoder(부호화하는 기계)

<br>

## 2. 구조

- Autoencoder는 기본적으로 3개의 층으로 구성

| 구성요소 | 설명 | 비유 |
| --- | --- | --- |
| Encoder | 입력 데이터를 저차원(latent space)으로 압축 | 큰 그림을 요약하는 단계 |
| Latent Space(잠재 공간) | 핵심 정보만 담은 압축된 표현 | 책의 목차처럼 중요한 정보만 남긴 것 |
| Decoder | 압축된 표현으로 원래 데이터를 복원 | 요약본으로 원래 책 내용을 다시 복원 |

```plaintext
입력 데이터 (Input)
       ↓
[Encoder]
       ↓
잠재 벡터 (Latent Vector, z)
       ↓
[Decoder]
       ↓
복원된 데이터 (Reconstructed Output)
```

<br>

## 3. 작동 원리

1. 이미지, 문장, 시계열 등을 입력 받음

2. Encoder로 압축 
    - Dense Layer 또는 CNN 등을 통해 차원을 줄임
    - 예 784차원(28X28 이미지) -> 32차원

3. Latent Vector 저장
    - 이건 "요약 정보"만 담은 압축된 표현

4. Decoder로 복원
    - 다시 그 정보를 가지고 원래처럼 복원 시도

5. 원래 입력과 얼마나 유사한지 비교
    - `Loss function`으로 차이 측정
        - loss = || 입력 - 출력 ||^2  # 재구성 오차

<br>

## 4. 목적과 활용

- 압축 과정 : 데이터의 핵심 구조만 뽑아냄
- 복원 과정 : 내가 제대로 핵심을 파악했는지 확인
- 비유 : 원래 300페이지짜리 책 핵심만 요약한 후 (Encoder), 그 요약만 가지고 다시 원래 내용 복원했을 때, 복원이 잘되면 내가 책의 핵심을 잘 이해한 것
- **의미 있는 압축**

| 목적 | 설명 | 예시 |
| --- | --- | --- |
| 차원 축소 | PCA보다 비선형 구조도 표현 가능 | 이미지 압축 |
| 노이즈 제거 | 입력에 노이즈가 있어도 깨끗하게 복원 | 손상된 사진 복원 |
| 이상 탐지 | 이상한 입력은 복원이 잘 안됨 -> 탐지 가능 | 신용카드 부정거래 탐지 |
| 생성 모델의 기초 | `VAE` 등으로 확장 가능 | 이미지 생성 |

- ### VAE(Variational Autoencoder)

    - 확률 기반의 Autoencoder
        - 기존 Autoencoder는 하나의 고정된 점으로 압축
        - 데이터를 하나의 점이 아니라 정규분포로 압축
    
    | 항목 | 일반 | VAE |
    | --- | --- | --- |
    | Latent | 고정된 벡터 z | 확률 분포 z~N(μ,σ^2) |
    | 하나의 대표값 | 여러 개의 가능성 중 샘플링 |
    | 목적 | 복원 잘하기 | 복원 + 생성도 잘하기 |
    | 학습 | 단순 MSE | MSE + KL Divergence |

    - #### 구조
    ```plaintext
    입력 x
    ↓
    [Encoder]
    → 평균(μ), 표준편차(σ) 출력
    ↓
    [Sampling z ~ N(μ, σ)]
    ↓
    [Decoder]
    ↓
    출력 x_hat
    ```

<br>

## 5. 코드 예시 (PyTorch)

```python
import torch.nn as nn

# Autoencoder 클래스 정의
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()  # nn.Module의 초기화

        # Encoder 정의: 입력 784 → 128 → 32차원
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),   # 784차원 (28x28 이미지) → 128차원
            nn.ReLU(),             # 활성화 함수: 비선형성 추가
            nn.Linear(128, 32),    # 128차원 → 32차원 (잠재 공간)
            nn.ReLU()              # 다시 비선형성 추가
        )

        # Decoder 정의: 32 → 128 → 784차원 (원래 이미지 복원)
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),    # 잠재 공간 → 중간 표현
            nn.ReLU(),             
            nn.Linear(128, 784),   # 다시 원래 차원으로 복원
            nn.Sigmoid()           # 0~1 범위로 정규화 (이미지 출력용)
        )

    # 순전파 함수: 입력 x → encoder → decoder → 출력
    def forward(self, x):
        z = self.encoder(x)        # 인코더를 통해 잠재 표현 추출
        return self.decoder(z)     # 디코더로 복원된 데이터 출력
```

<br>

## 6. Autoencoder vs PCA

| 항목 | Autoencoder | PCA |
| --- | --- | --- |
| 차원축소 | 가능 | 가능 |
| 비선형 표현 | 가능(비선형 함수 사용) | 불가능(선형) |
| 학습 방식 | 비지도 학습, 학습 필요 | 수학적 계산으로 구함 | 
| 표현력 | 훨씬 유현 | 제한적 |

