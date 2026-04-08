# ChannelGate: SENet에서의 영감과 설계 동기

## 1. SENet 핵심 복습

SENet (Squeeze-and-Excitation Networks, Hu et al. CVPR 2018)은 CNN의 채널 간 관계를 명시적으로 모델링하기 위한 구조다.
기존 Conv 연산은 채널 간 의존성을 암묵적(implicit)으로만 학습하는데,
SENet은 이를 **Squeeze → Excitation → Scale** 3단계로 명시화한다.

```
U (H×W×C)
    │
    ▼
[Squeeze]  GAP: z_c = (1/HW) ΣΣ u_c(i,j)        → z ∈ R^C
    │
    ▼
[Excitation]  s = σ(W₂ · ReLU(W₁ · z))           → s ∈ (0,1)^C
    │         (bottleneck FC: C → C/r → C)
    ▼
[Scale]  ũ_c = s_c · u_c                           → Ũ (H×W×C)
```

**핵심 철학:** 공간(H×W)을 collapse해서 채널별 global 통계를 뽑고,
그 통계로부터 "어떤 채널이 이 입력에서 중요한가"를 학습한다.

---

## 2. VLM Motion 문제에서의 병렬 관찰

VLM의 방향 인식 실패를 진단한 결과, 다음이 확인되었다:

- CLIP feature에 대한 linear probing → **100% accuracy** (방향 정보는 존재함)
- VLM end-to-end → **방향 분류 실패** (VLM이 그 정보를 사용하지 못함)

추가 분석에서:

> Projector output (3584d)의 채널 중 일부는 appearance (텍스처, 색상)를 인코딩하고,
> 일부는 motion (경계 변화, 시간적 패턴)을 인코딩한다.

즉 **"어떤 채널이 변화하는가"** 라는 정보가 채널 축에 분산되어 있고,
LLM은 이를 구별하지 못한 채 appearance-heavy 채널에 편향된다.

이 관찰이 SENet의 문제 설정과 정확히 대응된다:

| | SENet | ChannelGate |
|---|---|---|
| 문제 | Conv 채널 간 관계가 암묵적 | Motion 채널과 Appearance 채널이 혼재 |
| 목표 | "이 이미지에서 중요한 채널" 선택 | "이 비디오에서 변화하는 채널" 선택 |
| 해결 | Spatial squeeze → channel gate | **Temporal squeeze → channel gate** |

---

## 3. SENet → ChannelGate: 설계 변환

### 3.1 Squeeze: Spatial → Temporal

SENet의 squeeze는 공간 정보를 날려 채널별 global 통계를 만든다.

```python
# SENet squeeze
z_c = (1 / H*W) * sum_{i,j} u_c(i, j)   # 공간 평균 → 채널별 대표값
```

비디오에서 "중요한 채널"의 기준은 **시간에 따라 얼마나 변화했는가**다.
따라서 squeeze 축을 공간(H×W)에서 **시간(T)**으로 전환한다.

```python
# ChannelGate squeeze
delta = features[1:] - features[:-1]          # (T-1, N', D): 시간 변화량
delta_squeeze = delta.mean(dim=(0, 1))         # (D,): 채널별 평균 temporal 변화
```

`delta_squeeze[c]`가 크다 = c번째 채널이 시간에 따라 많이 변했다 = motion-relevant 채널.

### 3.2 Excitation: Sigmoid → 1 + α·tanh

SENet은 sigmoid를 써서 gate ∈ (0, 1)을 만든다.
이는 **불필요한 채널을 0에 가깝게 억제**하는 competitive suppression 전략이다.

```python
# SENet excitation
s = sigmoid(W₂ · ReLU(W₁ · z))    # ∈ (0, 1): 억제 가능
```

ChannelGate의 목적은 억제가 아닌 **motion 채널의 증폭**이다.
또한 frozen pretrained backbone 위에 올라가므로, appearance 채널을 죽이면
"빨간 공이 오른쪽으로" 같은 기술이 불가능해진다.

따라서 gate 범위를 1 중심으로 재설계한다:

```python
# ChannelGate excitation
gate = 1 + α · tanh(MLP(delta_squeeze))    # ∈ (1-α, 1+α): 증폭 중심
```

| | SENet | ChannelGate |
|---|---|---|
| gate 범위 | (0, 1) | (1-α, 1+α) |
| 전략 | 불필요 채널 억제 | motion 채널 증폭 |
| zero-init | 불가 (`σ(0) = 0.5`) | 가능 (`α=0 → gate=1`) |

### 3.3 Zero-Init: Identity로 시작

SENet은 처음부터 모든 채널을 0.5배로 줄이며 시작한다.
ChannelGate는 `α=0`으로 초기화하여 **처음엔 정확히 identity**로 동작한다.

```python
# α=0일 때:
gate = 1 + 0 · tanh(MLP(delta_squeeze)) = 1
output = features * 1 = features    # pretrained baseline과 완전히 동일
```

학습이 진행되며 α가 커지고, gate가 점진적으로 motion 채널을 강조한다.
이는 ControlNet의 zero-init residual과 같은 "안전한 시작점" 전략이다.

### 3.4 Scale: Channel-wise Multiply (동일)

```python
# SENet
ũ_c = s_c · u_c

# ChannelGate
output = features * gate    # (T, N', D) * (D,) broadcast
```

채널별 scalar 곱으로 feature를 recalibrate하는 방식은 SENet과 동일하다.

---

## 4. 전체 구조 대응 요약

```
           SENet                          ChannelGate
─────────────────────────────────────────────────────────────
Input    U (H×W×C)                   features (T, N', D)

Squeeze  GAP over (H, W)             delta.mean over (T-1, N')
         → z ∈ R^C                   → delta_squeeze ∈ R^D
         "공간을 날려 채널 통계"      "시간 변화량으로 채널 통계"

Excite   σ(W₂·ReLU(W₁·z))           1 + α·tanh(MLP(delta_sq))
         → s ∈ (0,1)^C               → gate ∈ (1-α, 1+α)^D
         "채널 중요도 (억제 포함)"    "motion 채널 증폭 (억제 없음)"

Scale    ũ_c = s_c · u_c             output = features * gate
         "recalibrated features"      "motion-enhanced features"
─────────────────────────────────────────────────────────────
목적     불필요 채널 제거              motion 채널 증폭
task     이미지 분류                  비디오 방향 인식
```

---

## 5. v2~v4 대비 차별점

기존 motion query 방법(v2~v4)은 auxiliary loss만으로 projector를 간접 학습시켰다.
ChannelGate는 **LLM이 실제로 받는 feature token 자체를 수정**한다는 점에서 근본적으로 다르다.

```
v2~v4:  features → LLM
        ↑ (auxiliary loss로 간접 영향)

ChannelGate:  features → [gate recalibration] → LLM
              ↑ LLM input 자체가 달라짐
```

SENet이 "conv 다음 layer가 보는 feature"를 recalibrate하듯,
ChannelGate는 "LLM이 보는 visual token"을 recalibrate한다.

---

## 6. 파라미터 규모

SENet bottleneck과 동일한 구조: `D → D/r → D`

```
D = 3584, r = 16
3584 → 224 → 3584
params ≈ 3584×224 + 224×3584 ≈ 1.6M
```

전체 LLaVA-OneVision (7B) 대비 **0.023%** 증가.
