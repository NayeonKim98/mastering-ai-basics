# 1. Transformer란?

- 2017년 논문 "Attention is All You Need"에서 처음으로 소개된 모델

- 입력을 받고 출력을 만들어내는 인코더-디코더 구조의 모델 (다른 형태로 변형)

- 주로 자연어처리(NLP, Natural Language Processing)에서 문장을 번역하거나 요약하거나 글을 생성할 대 쓰임

<br>

# 2. 전통 모델과의 차이

- 기존에는 RNN이나 LSTM같은 순차적 모델을 사용
- 하지만 RNN은 순서대로 계산하여 병렬화(parallelization)가 어려웠고, 오래된 정보를 기억하는데 약했음

#### Transformer는?

- 완전 병렬 처리 가능
- 긴 문장도 한 번에 다룰 수 있음
- Attention 메커니즘만으로 전체를 처리

#### 비유

- RNN이 마치 기차 칸칸이 연결된 기차라면,
- Transformer는 비행기처럼 한 번에 쫙! 전 구간을 뛰어넘는다.

<br>

# 3. Transformer 구조

```markdown
[ 입력 문장 ]
    ↓
[ Encoder Stack ]
    ↓
[ Context 벡터 ]
    ↓
[ Decoder Stack ]
    ↓
[ 출력 문장 ]
```

| Encoder | Decoder |
| --- | --- |
| 입력을 처리해 정보 요약본 생성 | 요약본을 받아 출력 문장 생성 |
| Self-Attention 사용 | Self-Attention + Encoder-Decoder Attention 사용 |
| 여러 층(layer)을 쌓음 | 여러 층(layer)을 쌓음 |

<br>

# 4. Transformer 핵심 구성 요소

<br>

## Attention Mechanism (주의 메커니즘)

- 문장 안의 단어들이 서로 얼마나 중요한지 **스스로 판단**하는 것
- 예: "나는 사과를 먹었다." 문장에서,
    - "먹었다"를 해석할 때 '사과'가 중요. 그걸 attention이 파악

#### Self-Attention(자기-주의)
    - 한 문장 안에서 단어들끼리 서로 주목하는 것!

<br>

## Multi-Head Attention

- 하나의 Attention만 보는 게 아니라, 여러 방향으로 동시에 보는 것
- 서로 다른 관점으로 정보를 본다.

<br>

## Position Encoding

- Transformer는 RNN처럼 순서를 기억하는 구조가 아님
- 각 단어의 위치를 따로 인코딩해서 넣어줘야함

- #### Encoding
    - 정보를 다른 형태로 바꿔 저장하는 것
    - 책의 페이지 번호처럼, 몇 번째 단어인가를 알려주는 시스템

<br>

## Feed Forward Network (FFN)

- 각 단어별로 독립적으로 처리하는 작은 MLP(다층 퍼셉트론, Multi-Layer Perceptron)층
- 미니 신경망

<br>

## Residual Connection & Layer Normalization

- Residual Connection(잔차 연결) : 입력 + 출력 = 정보 손실을 막아줌

- Layer Normalization : 학습을 안정시키기 위해 각 층별로 분포를 정리

<br>

# 5. Transformer가 혁신적이었던 이유

| 기존 방법 | Transformer |
| --- | --- |
| 순차처리(RNN)로 느림 | 병렬 처리 |
| 긴 문장 다루기 어려움 | 긴 문장 문제 해결 |
| 기억력 문제 | Attention으로 강력한 기억 가능 |

### Transformer의 흐름

```mathematica
Input
 └▶ Positional Encoding
 └▶ Encoder Layer (Multi-head Attention → Feed Forward) × N
 └▶ Encoder Output

Decoder Input
 └▶ Positional Encoding
 └▶ Decoder Layer (Masked Multi-head Attention → Encoder-Decoder Attention → Feed Forward) × N
 └▶ Output Prediction
```
