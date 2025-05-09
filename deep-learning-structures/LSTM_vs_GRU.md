# LSTM vs GRU

## 1. 공통된 목표

- 둘 다 기본적으로 RNN(Recurrent Neural Network) 계열. 즉, `순서가 있는 데이터(Sequence Data)`를 처리하려고 만듦

    - #### 순서가 있는 데이터 
        - 문장, 주식 가격, 음악 음파 등

- "과거 정보를 기억하고(long-term memory) 현재 예측에 활용하자."

- BUT 기존 RNN은 장기 의존성 문제, 즉 오래된 정보를 잘 기억하지 못함. 시간이 오래 지나면 기억이 희미해지는 문제가 생김.

- 그래서 LSTM과 GRU가 등장!

## 2. LSTM (Long Short-Term Memory)

### 구조

- Forget Gate : 과거 정보를 얼마나 잊을지 결정
- Input Gate : 새로 들어온 정보를 얼마나 기억할지(store) 결정
- Cell State : 메인 기억 저장소
- Output Gate : 다음 hidden state에 무엇을 출력할지 결정

### 동작 과정

1. Forget Gate
    - 과거 기억 중 버릴 것을 결정(예: 오래된 기억, 별로 중요하지 않은 기억은 버림)
2. Input Gate
    - 새로운 정보 중 저장할 것을 결정
3. Cell State 업데이트
    -  (과거 기억 x forget) + (새 정보 x input)
4. Output Gate
    - 현재 step의 최종 결과

#### 비유

- 다이어리를 들고 다니는 사람
    - 매일매일 다이어리에 중요한 일만 기록,
    - 지나간 일 중 쓸모 없는건 과감히 지움
    - 그날그날 필요한 내용만 꺼낸다.

## 3. GRU (Gated Recurrent Unit)

### 구조

- Update Gate : 얼마나 과거를 유지하고, 얼마나 새로 업데이트할 지
- Reset Gate : 과거 정보를 얼마나 무시할지

* Cell State는 따로 없음!
* 모든 정보는 hidden state 하나에 통합

### 동작 과정

1. Reset Gate
    - 이전 hidden state 중 무시할 부분을 골라냄
2. Update Gate
    - 얼마나 과거 기억을 유지하고, 얼마나 새로 쓸지 결정
3. 최종 hidden state 계산
    - Reset을 적용한 과거 기억 + 새로운 입력 정보를 조합

#### 비유

- 스마트폰 메모 앱을 쓰는 사람
    - 과거 메모를 완전 삭제하거나 업데이트할 때
    - 별도의 다이어리(=cell state) 없이 메모장 하나로 전부 관리

## 4. LSTM vs GRU 비교

| 항목 | LSTM | GRU |
| --- | --- | --- |
| 구조 복잡성 | 더 복잡(게이트 3개 + 셀 상태) | 더 단순(게이트 2개) |
| 파라미터 수 | 많음(무거움) | 적음(가벼움) |
| 학습 속도 | 느릴 수 있음 | 빠름 |
| 기억 능력 | 긴 기억에 더 강함 | 짧은 기억에 적합 |
| 적용 사례 | 긴 문장 번역, 장기 예측 | 빠른 응답이 필요한 시스템, 간단한 모델링 |
| 비유 | 다이어리에 기록하는 사람 | 메모앱 쓰는 사람 |

- LSTM은 아주 긴 시퀀스를 다룰 때 더 성능이 좋다. (예: 긴 소설 요약, 장기 주가 예측)
- GRU는 LSTM보다 연산량이 적어 빠르다.(예: 특히 모바일 기기, 작은 모델에 유리)
