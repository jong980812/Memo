# Method Proposals: Extending Delta-Direct for Better In-LLM Direction Flow

> 기반: `analysis/token_linear_probing_R2R_1500/FINAL_REPORT.md`
> 목적: LLM 내부에서 direction 정보가 **last token까지 효과적으로 흐르도록**, 특히 direction→letter 매핑이 더 일찍/강하게 학습되도록 하는 method 제안

---

## 1. 분석으로부터 도출된 Design Principles

### Fact 1: Vision token에 direction 정보는 이미 있다
- Delta direction probe = 0.87 (pre-proj), 0.89 (post-proj) — linearly decodable
- → **정보 부족이 아니라 "활용" 문제**

### Fact 2: Vanilla LLM도 L1–L4에서 direction 흡수함 (0.68)
- Layer 4가 peak
- → 얕은 layer 추출은 문제 아님

### Fact 3: L8 이후 vanilla direction 감쇠 (0.68 → 0.54 at L28)
- Baseline/Ours는 L20+에서 0.84 유지
- → **Deep layer 유지가 fine-tuning의 핵심 효과**

### Fact 4: Letter mapping은 오직 L28에서만 점프 (0.25 → 0.47–0.53)
- 중간 layer에선 letter probe 모두 chance 근방
- → "direction 정보 → letter 출력" 변환이 **LM head 바로 앞에서 급격히** 일어남

### Fact 5: Delta-Direct가 letter mapping 최강 (0.528 vs baseline 0.473)
- Direction regression loss (projector output 단계)가 letter head 학습까지 간접 기여
- → Explicit direction supervision은 **multi-location 확장**하면 더 강할 듯

### Fact 6: 큰 weight 변화 필요 없음 (ours 5% rel_diff)
- Small intervention, big effect
- → **경량 module** (LoRA-scale parameters) 충분

---

## 2. Target 위치 (어디에 개입할 것인가)

분석 결과를 종합하면 개입 후보는 3곳:

| 위치 | 현재 상태 (vanilla) | 문제 |
|---|---|---|
| **Projector output** | Direction F-stat 443 | 이미 delta_direct가 여기 개입 |
| **LLM middle (L8–L20)** | Direction 0.62–0.68 (decay 중) | ★ **Unused target** — direction 감쇠 방지 가능 |
| **LLM late (L24–L28)** | Letter mapping 부재 | Letter 학습 직접 강화 가능 |

**→ 제안 method들은 Middle layer 혹은 Middle + Late에 개입**

---

## 3. 6가지 Method Proposal

### 3.1 Idea 1: **LD³ (Layer-wise Delta-Direct)** ★ 가장 자연스러운 확장

#### Motivation
Delta_direct은 projector output에 한 번만 direction loss를 건다. 하지만 LLM 내부에서 direction이 L8 이후 감쇠 → **각 layer에서도 direction 정보를 유지**하도록 supervisory signal을 여러 layer에 뿌리자.

#### Architecture
```
Projector out  →  LLM L0 L1 ... L8 ... L16 ... L28
                                ↓      ↓       ↓
                             pool    pool    pool
                            (vis toks only)
                                ↓      ↓       ↓
                          Δ(last-first)  ... (for each)
                                ↓      ↓       ↓
                        DirHead_8  DirHead_16  DirHead_28
                                ↓      ↓       ↓
                             L_dir_8 + L_dir_16 + L_dir_28
```

각 supervision layer `l ∈ {8, 16, 28}`에서:
1. LLM hidden state에서 **vision token 위치만 추출** (T frames × N_pool tokens)
2. Frame 단위로 mean-pool → (T, D_llm)
3. `delta_l = pooled[-1] - pooled[0]` → (D_llm,)
4. `DirHead_l(delta_l)` → 방향 예측 (regression on unit vector or classification)
5. Loss: `L_dir_l = ||DirHead_l(delta_l) - dir_gt||²` or cross-entropy

Total loss: `L = L_lm + λ_proj · L_dir_proj + Σ_l λ_l · L_dir_l`

#### Expected effect
- **L8–L20에서 direction decay 방지** → last token에 강한 신호 도달
- Deep layer probe에서 direction ↑, **letter mapping도 덩달아 ↑ 예상**
- Baseline의 "deep layer direction 보존" + delta_direct의 "explicit supervision" 둘 다 얻음

#### Hyperparameters / Engineering
- Supervision layers: 먼저 {L8, L16} 두 곳으로 시작; 효과 있으면 {L4, L12, L20}로 확장
- Head architecture: 2-layer MLP (D_llm → 512 → 2 for (dx, dy))
- Loss weight: `λ_l = 0.1` 수준 (LM loss dominate 방지)
- Gradient flow: LLM backbone LoRA params로 흘러가야 함

#### Pros
- Analysis findings와 직접 연결 (L8 decay 지점에 supervision)
- Delta_direct의 자연스러운 generalization
- Layer별 효과를 ablation으로 분리 가능

#### Cons
- Supervision layers 많아지면 training 느려짐
- 어느 layer 조합이 best인지 탐색 필요

---

### 3.2 Idea 2: **Last-Token Direction Anchor** ★ 가장 target specific

#### Motivation
Analysis에서 **letter mapping은 오직 L28에서만** 나타남. 그렇다면 **last token hidden state 자체**에 direction supervision을 거는 건 어떤가?

Vanilla L20의 last token direction probe = 0.64 → 0.84로 올라갈 수 있다면, L28 letter mapping도 자연히 올라갈 가능성.

#### Architecture
```
Training time:
  각 step에서 LLM forward → at specific layers {L12, L20},
  last_token_hidden = hs[l][0, -1, :]
  dir_pred = DirHead_l(last_token_hidden)
  L_dir_last_l = || dir_pred - dir_gt ||²
```

Inference time:
- 이 loss로 튜닝된 모델로 그냥 MCQ generation (추가 module 없음)

#### Expected effect
- **Last token이 layer 12–20에서부터 direction-readable form이 되도록 강제**
- Letter head (L28→vocab)가 더 쉽게 letter를 고르게 됨
- Information-generation gap 축소 (vanilla 0.54/0.25 → 예상 0.85/0.60+)

#### Comparison with Idea 1
- LD³: vision token pooling 기반 → 영상 content 중심
- Anchor: last token 기반 → LM 출력 측면

둘 다 같이 쓰는 것도 가능 (complementary).

#### Pros
- 극도로 target specific (letter mapping 직접 타겟)
- 구현 단순 (head 1–2개만 추가)

#### Cons
- Last token은 prompt의 위치라서 vision info 외 text 정보도 섞여 있음 → supervision이 혼선 줄 수 있음
- Candidate shuffle 때문에 last token은 MCQ format-specific — robustness 우려

---

### 3.3 Idea 3: **Direction Residual Stream** ★ 구조적으로 우아

#### Motivation
LLM이 깊어지면서 direction 정보가 감쇠. 이 **감쇠를 막기 위해 얕은 layer의 direction representation을 깊은 layer에 skip connection으로 주입**.

L4 peak (direction 0.68) 가 있는데 L28에 가면 0.54로 떨어진다. 이를 **residual add**로 보존.

#### Architecture
```
Pooled vision-token summary at projector output: v_summary ∈ R^3584
                                  ↓
                            DirProj(v_summary)
                                  ↓
                          direction_embedding ∈ R^3584
                                  ↓
At specific layer L_target (e.g., L16):
  hs_new[l = L_target] = hs[l] + α · direction_embedding  (broadcast over seq_len)
```

또는 last token position에만 inject:
```
hs_new[l, -1, :] = hs[l, -1, :] + α · direction_embedding
```

`DirProj`는 학습 가능한 small MLP. `direction_embedding`은 영상별로 하나.

#### Expected effect
- Direction info가 shallow layer에서 deep layer로 **보존된 채로 전달**
- L16 이후 direction probe acc 유지
- **Delta_direct 학습 때 함께 학습하면**: DirProj의 파라미터가 direction 정보를 "distilled" form으로 유지하도록 학습됨

#### Hyperparameters
- α: trainable scalar (learnable gate), 초기값 0.1
- Injection layers: L12 혹은 L16 (decay 시작 지점)

#### Pros
- 구조적 우아함 (ResNet 감성)
- 추가 loss 없이 architecture 개선만으로 효과 가능
- Delta_direct과 orthogonal → combine 가능

#### Cons
- Implementation 복잡 (LLM forward hook 필요)
- α 튜닝 민감

---

### 3.4 Idea 4: **Direction-Conditional Letter Head (Two-Stage Decoding)**

#### Motivation
Analysis에서 direction과 letter가 다르게 학습됨. **명시적으로 두 단계로 분해**:
1. 먼저 direction을 예측 (8-way)
2. 그 direction을 candidates에 매칭해 letter 출력

#### Architecture (training)
```
LLM forward → hs[L20] (or similar depth)
            → direction_head(last_token_hs[L20])  ← aux loss (supervised by dir_gt)
            → direction_embedding (learnable) fed back to LLM at L21+
            ↓
LLM remaining layers with direction context
            → letter decoded at L28
```

혹은 training time only:
- L20에서 direction 예측하게 학습
- Inference는 full generation (no change)

#### Expected effect
- LLM이 "direction 이해 → letter 출력"을 **명시적 파이프라인으로 학습**
- Letter mapping 학습이 direction loss에 의해 "guided"

#### Pros
- 사람이 이해하기 쉬운 pipeline
- Intermediate accuracy 측정 가능 (diagnostic value)

#### Cons
- Two-stage training 복잡
- Inference 때 직접적 변화 없으면 효과 제한적

---

### 3.5 Idea 5: **Layer-wise Contrastive Direction**

#### Motivation
Direction이 같은 영상은 last token hidden이 비슷하고, 다른 방향 영상은 달라야 한다. 이를 **contrastive loss**로 강제.

#### Architecture (training)
```
Batch 내에서 4 방향 영상들 섞여있음.
At each supervision layer l ∈ {8, 16, 24}:
  hs_last[l] ∈ R^{B, 3584}  (batch of last token hidden)
  L_contrastive_l = SupConLoss(hs_last[l], direction_labels)
```

SupConLoss (Supervised Contrastive Loss):
- Positive pairs: same direction
- Negative pairs: different direction

#### Expected effect
- Last token representation이 **direction-aware manifold**에 놓임
- Letter head가 이 structured space에서 쉽게 방향→letter 매핑

#### Pros
- Direction-specific supervision 없이도 structure 유도
- Proven effective in vision tasks

#### Cons
- Batch size 커야 효과적 (4 방향 모두 포함된 batch)
- Hyperparameter sensitive (temperature 등)

---

### 3.6 Idea 6: **Motion Token Injection** (explicit direction channel)

#### Motivation
현재 vision tokens는 patch-wise 정보만 담음. **motion/direction**을 명시적 token으로 LLM에 주입.

#### Architecture
```
Video frames (T, 729, D_vis)
      ↓ mean pool per frame
(T, D_vis)
      ↓ compute delta features
delta_vec = [f[-1] - f[0], or Σ(f[t+1]-f[t]), or LSTM(f)]
      ↓ project to LLM dim
motion_token ∈ R^{1, 3584}
      ↓ prepend or insert between vision tokens and question text
<vision_tokens> [MOTION] <question>
      ↓ LLM processes with explicit motion signal
```

Training with direction regression on `motion_token` (consistency with delta_direct).

#### Expected effect
- LLM은 motion을 별도 "attention target"으로 사용 가능
- Direction info가 vision token에 얇게 퍼진 게 아니라 집중된 한 token에 존재 → attention 더 쉽게 pickup
- Letter head가 이 token을 읽어 answer 생성

#### Pros
- 명시적, interpretable
- Delta_direct의 자연스러운 확장 (direction vector → direction token)

#### Cons
- Sequence length 증가 (경미)
- Motion token이 무시될 수도 (attention이 학습되지 않으면)

---

## 4. 우선순위 및 조합 전략

### Tier 1 (가장 먼저 시도)

1. **LD³ (Layer-wise Delta-Direct)** — 3.1
   - Rationale: Analysis findings와 가장 직접 연결
   - 단순하고 parameter 거의 안 늘어남
   - Delta_direct의 straightforward extension

2. **Last-Token Direction Anchor** — 3.2
   - Rationale: Letter mapping을 직접 타겟
   - 구현 제일 간단

### Tier 2 (Tier 1이 성공하면 다음)

3. **Direction Residual Stream** — 3.3
   - Rationale: 구조적 개선, Tier 1과 합쳐서 synergy

4. **Motion Token Injection** — 3.6
   - Rationale: 완전히 다른 각도의 접근, 실패해도 배울 점 많음

### Tier 3 (실험적)

5. **Direction-Conditional Letter Head** — 3.4
6. **Layer-wise Contrastive** — 3.5

### 조합 제안

- **Delta_direct (기존) + LD³ (L8, L16) + Anchor (L20)**:
  - 3 depth에서 direction supervision (projector / middle / late)
  - Ablation으로 각 layer 기여도 분리 가능
- **Delta_direct + Direction Residual**:
  - Supervisory loss + structural aid 둘 다

---

## 5. Validation Plan

각 method에 대해:

### A. Representation-level (이번 analysis와 동일)
1. LLM hidden probe (direction, gt_letter) at L{4, 12, 20, 28}
2. 기대: direction probe 0.80+ at L20, letter probe 0.55+ at L28

### B. Generation-level
1. MCQ accuracy on R2R_4way_1500 eval set
2. 기대: vanilla 25% → baseline 45% → ours 55% → proposed 60+%

### C. Ablation
1. Supervision layer set 변경 (L8 only, L16 only, L8+L16, etc.)
2. Loss weight tuning
3. Compare Tier 1 individual vs combined

---

## 6. Minimal Code Changes

LD³ 예시:

```python
# In LlavaMetaModel (llava_arch.py):
class LayerwiseDeltaDirectModule(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, layer_indices=(8, 16)):
        super().__init__()
        self.layer_indices = layer_indices
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 512), nn.GELU(), nn.Linear(512, 2))
            for _ in layer_indices
        ])
    
    def compute_loss(self, hidden_states, vision_token_mask, direction_gt, frame_indices):
        # hidden_states: tuple from output_hidden_states=True
        # vision_token_mask: which positions are vision tokens
        # frame_indices: which frames each vision token belongs to
        losses = {}
        for i, l in enumerate(self.layer_indices):
            hs_l = hidden_states[l]  # (B, seq, D)
            # Extract vision token positions, group by frame, mean-pool
            vis_hs = extract_and_pool_by_frame(hs_l, vision_token_mask, frame_indices)  # (B, T, D)
            delta = vis_hs[:, -1, :] - vis_hs[:, 0, :]
            dir_pred = self.heads[i](delta)  # (B, 2)
            losses[f"dir_l{l}"] = F.mse_loss(dir_pred, direction_gt)
        return losses
```

Training loop에 추가:
```python
outputs = model(input_ids, images=..., output_hidden_states=True)
lm_loss = outputs.loss
aux_losses = lld3_module.compute_loss(outputs.hidden_states, vt_mask, dir_gt, frame_idx)
total_loss = lm_loss + sum(0.1 * l for l in aux_losses.values())
```

---

## 7. Open Questions (실험으로 답할 것)

1. **어느 layer가 가장 effective한가?** L8? L16? L20?
2. **Projector-only vs In-LLM supervision 어느 게 더 강한가?** Delta_direct vs Anchor alone.
3. **Contrastive vs Regression 어느 게 더 잘 일반화하나?**
4. **Training data complexity (shape_simple vs obj_place)와 method 효과의 관계는?**
5. **Diagonal direction (8-way MCQ)에서도 이 method들이 scale되나?**

---

## 맺음말

이번 분석의 핵심은 **"information is there, mapping is the bottleneck"**. 따라서 method 설계 방향은:

1. **새 정보를 만들지 말고**, 이미 있는 정보가 **deep layer까지 손실없이 전달**되도록
2. **Letter mapping 학습을 직접 강화**하거나, **auxiliary supervision**을 여러 layer에 분산
3. **Delta_direct의 성공 공식**(direction regression loss) 을 **multi-location**으로 복제

**Tier 1 첫 실험 추천**: LD³ with {L8, L16} + 기존 delta_direct loss → MCQ 정확도 비교
