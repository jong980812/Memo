# ChannelGate 구조적 한계 분석

## 한계 1. Gate가 video-level 단일 벡터

```python
delta = features[1:] - features[:-1]      # (T-1, N', D)
delta_squeeze = delta.mean(dim=(0, 1))    # ← (D,) 단일 벡터로 collapse
gate = 1 + α·tanh(MLP(delta_squeeze))    # (D,)
output = features * gate                  # 모든 T 프레임에 동일 gate 적용
```

**문제:** T-1개의 delta를 모두 평균내어 단일 벡터로 만들기 때문에,
temporal dynamics 정보가 완전히 사라진다.

- "처음엔 빠르게 → 나중엔 멈춤" 같은 시간에 따른 motion 변화를 표현 불가
- 프레임 `t=0`과 `t=10`이 동일한 gate를 받음
- motion이 **언제** 일어나는지를 모름

---

## 한계 2. Delta mean이 방향 부호를 날림 ← 방향 연구에서 치명적

```python
delta_squeeze = delta.mean(dim=(0, 1))
```

- 오른쪽으로 움직이는 물체: `delta_c > 0`
- 왼쪽으로 움직이는 물체: `delta_c < 0`
- 둘이 반반 섞이면: `delta_c ≈ 0` → `gate ≈ 1` (아무것도 안 함)

더 심각한 문제: **오른쪽 moving과 왼쪽 moving이 동일한 gate 값을 만들 수 있음.**

gate는 "motion이 있는 채널"을 증폭하지만, 그 motion이 어느 방향인지는
원천적으로 encoding하지 않는 구조다.
방향 분류가 핵심 task인 본 연구에서 핵심 signal이 소실된다.

---

## 한계 3. First-frame bias 문제를 근본적으로 못 건드림

```python
output = features * gate   # 첫 프레임도 동일한 gate 적용
```

진단 실험의 핵심 결과: VLM은 첫 프레임에 anchoring되어 이후 프레임의
motion을 무시하는 경향이 있다. 그러나 ChannelGate는:

- 첫 프레임의 강한 appearance prior를 **억제하는 메커니즘이 없음**
- gate가 motion channel을 증폭하더라도, LLM이 여전히 첫 프레임의
  content token에 attention을 몰아줄 수 있음
- first-frame anchoring의 근본 원인(LLM의 attention 편향)을 건드리지 않음

---

## 한계 4. LM loss만으로 gate가 실제로 motion channel을 찾는다는 보장 없음

```python
# docs에 "optional direction probe"라고 명시됨
# gate 자체는 LM loss로 학습
loss = LM_loss
```

LM loss는 "다음 토큰 예측"이 목적이므로, gate가 motion channel을 찾는 게 아니라
**language modeling에 유리한 channel**을 찾을 수 있다.

- gate가 실제로 temporal change에 민감한 채널을 선택하는지 검증 불가
- direction supervision 없이는 gate 학습 방향이 보장되지 않음
- "motion channel을 증폭한다"는 설계 의도가 실제로 실현되는지 확인할 수단이 없음

---

## 한계 5. Spatial 정보 완전 소실

```python
delta.mean(dim=(0, 1))   # T-1, N' 모두 평균 → 공간 정보 없음
```

N' 차원(patch 위치)까지 평균내기 때문에:

- 화면 **왼쪽**에서 움직이는지 **오른쪽**에서 움직이는지 구별 불가
- 어떤 spatial region의 변화인지 정보 소실
- global한 채널 통계만 남음

방향 이해에 있어 "어디서 움직이는가(where)"는 "무엇이 변하는가(what channel)"만큼
중요한 정보임에도 설계상 접근 불가능하다.

---

## 요약

| 한계 | 심각도 | 이유 |
|---|---|---|
| Video-level 단일 gate | 중 | temporal dynamics 소실 |
| 방향 부호 소실 | **상** | direction task에서 핵심 signal이 없어짐 |
| First-frame bias 미해결 | **상** | 연구의 핵심 문제를 구조적으로 건드리지 않음 |
| LM loss만으로 학습 | 중 | gate가 motion channel 찾는다는 보장 없음 |
| Spatial 정보 소실 | 중 | where 정보 없음 |

**가장 치명적인 것은 한계 2와 3이다.**
Gate가 "motion이 있는 채널"은 찾을 수 있어도,
"어느 방향으로"는 원천적으로 구분 불가능한 구조이며,
VLM의 first-frame anchoring 편향에는 직접적으로 대응하지 못한다.
