# Vision Token Information Analysis on R2R_video_1500
### Why our methods (delta_direct, channel_gate) improve VLM direction understanding — a representation-level investigation

> **통합 최종 보고서 (R2R_video_1500 데이터 기준)**
> Dataset: R2R_video_1500 (4 conditions × 6000 videos), MCQ: R2R_4way_1500_json (4-way; UDLR shuffled)
> Primary condition: **obj_place** (real object + real background, 가장 현실적·어려운 조건)

---




# TL;DR (한 페이지 요약)

1. **Vision tokens carry rich spatial-temporal information.** SigLip ViT가 absolute position embedding + self-attention으로 729개 패치에 위치 정보를 전파하므로, **mean-pool(729→1) 후에도 D-dim 벡터에 위치가 보존**됨 (obj_place pre-proj R² = 0.81/0.87 for x/y).

2. **Position is encoded in a distributed way** — low-rank subspace가 아님. PCA top-50 dim (variance 92%)으로는 R²(x) = 0.03에 불과. 위치 정보는 **low-variance tail에 분산**. 또한 top-500 position-correlated dim을 제거해도 R²(x) = 0.41이 남음.

3. **Direction emerges from temporal position dynamics.** Mean-pooled vector 2개 차이(delta)만으로도 방향 87% 예측. 시간 순서 제거(T-mean)하면 chance 근처(0.32)로 붕괴 → **direction은 순전히 position dynamics에서 파생**.

4. **Position/Direction and Semantic live in disjoint subspaces.** Top-50 dim 기준 position ∩ object = 0–1/50, direction ∩ object = 0–1/50 (random 기대치 대비). Projector가 이 분리를 그대로 전달.

5. **Visual complexity (especially background) degrades direction encoding.** shape_color(단색, 도형)의 delta_dir 99% → obj_place(실배경, 실물체) 78%. F > 5 인 dim 수도 1084 → 610으로 43% 감소.

6. **Vanilla LLaVA는 vision info가 layer 1부터 last token에 흐르지만 direction → letter 매핑을 못 함.** L28에서 direction probe 0.58, letter probe **0.30** (chance 0.25 — **사실상 random**). 즉 "방향은 아는데 답변 letter를 못 고름".

7. **Baseline (LoRA only) changes projector by just 1–2% in weight, nearly matches vanilla in representation**, but significantly improves LLM deep-layer direction retention. L28 letter probe **0.649** (+35%p over vanilla).

8. **Our methods (delta_direct, channel_gate) apply a ~5% projector weight delta** (row cosine > 0.98 — 방향은 보존, 스케일만 미세 조정). Direction encoding strength, linear probe acc, 그리고 특히 **last-layer letter mapping이 크게 향상** — delta_direct letter L28 = **0.683** (+38%p over vanilla, +3.4%p over baseline). 또한 letter mapping이 **L20부터 일찍 출현** (≥ 0.50).

9. **Conclusion for method design**: 정보는 이미 있다 (vanilla에도). 문제는 **정보를 task-specific 답변 포맷(letter)으로 매핑**하는 마지막 단계. Explicit direction supervision (delta_direct) / direction-focused gating (channel_gate)이 projector와 LLM LoRA를 함께 튜닝해 이 매핑 격차를 줄인다.

10. **Trajectory geometry는 모델간 거의 동일** (Section I). Per-video curvature ~104° (노이즈 우세), class-mean curvature 36.5–37.8° (channel_gate 가장 straight, 차이 ~1°). Projector가 pre-proj 대비 trajectory를 ~4° 더 straight 하게 만들지만, fine-tuning은 추가 효과 미미 — **ours의 효과는 geometry 변화가 아니라 LLM-side mapping**.

---

# 목차

1. [연구 동기 및 핵심 질문](#1-연구-동기-및-핵심-질문)
2. [배경 지식](#2-배경-지식)
3. [데이터 및 모델](#3-데이터-및-모델)
4. [용어 정리 (Glossary)](#4-용어-정리-glossary)
5. [방법론: Linear Probing](#5-방법론-linear-probing)
6. **Experiments**
   - [A. Vision tokens carry position information](#a-vision-tokens-carry-position-information)
   - [B. Position is encoded in a distributed subspace](#b-position-is-encoded-in-a-distributed-subspace)
   - [C. Direction emerges from temporal dynamics](#c-direction-emerges-from-temporal-dynamics)
   - [D. Position/Direction vs Semantic — dim-level separation](#d-positiondirection-vs-semantic--dim-level-separation)
   - [E. Visual complexity degrades direction encoding](#e-visual-complexity-degrades-direction-encoding)
   - [F. Vanilla → Baseline → Ours — projector-level comparison](#f-vanilla--baseline--ours--projector-level-comparison)
   - [G. Projector weight change analysis](#g-projector-weight-change-analysis)
   - [H. LLM hidden state propagation — the information–generation gap](#h-llm-hidden-state-propagation--the-informationgeneration-gap)
   - [I. Trajectory straightness in representation space](#i-trajectory-straightness-in-representation-space)
7. [Synthesis: vanilla → baseline → ours 논리](#7-synthesis-vanilla--baseline--ours-논리)
8. [Method design implications](#8-method-design-implications)
9. [부록: Implementation Details (per experiment)](#9-부록-implementation-details-per-experiment)
10. [파일 구조](#10-파일-구조)

---

# 1. 연구 동기 및 핵심 질문

## 1.1 Problem Setting

Video LLM (LLaVA-Video)은 영상의 **이동 방향(direction)**을 답하는 MCQ에서 종종 낮은 정확도를 보인다. 우리가 제안한 `delta_direct`, `channel_gate` method가 이 문제를 개선하는 것은 알고 있지만,

> **Representation level에서 무엇이 어떻게 바뀌어서 개선되는가?**

이를 밝혀야 method의 정당성을 주장할 수 있다.

## 1.2 Research Questions

단계적으로 9개의 질문을 설정하고 각각 실험으로 답한다.

| # | 질문 | 답 섹션 |
|---|---|---|
| Q1 | SigLip ViT가 만드는 729개 패치 토큰을 mean-pool 한 뒤에도 위치 정보가 살아있나? | **A** |
| Q2 | 그 위치 정보는 특정 소수 dim에 집중? 전체 D에 분산? | **B** |
| Q3 | 방향 정보는? 어떻게 파생되고 얼마나 decodable? | **C** |
| Q4 | 위치/방향/Semantic 정보가 같은 dim을 공유하나, 아니면 분리? | **D** |
| Q5 | 시각적 복잡도(배경/물체)가 direction encoding에 어떤 영향? | **E** |
| Q6 | Fine-tuning(baseline vs ours)은 projector level에서 무엇을 바꾸나? | **F**, **G** |
| Q7 | LLM이 이 정보를 last token으로 얼마나 가져가는가? | **H** |
| Q8 | Vanilla vs trained 모델의 generation 차이는 어느 layer에서 발생? | **H** |
| Q9 | "Direction 정보를 가짐"과 "MCQ letter를 올바르게 출력함"이 일치하나? | **H** |

## 1.3 핵심 가설

1. **Equivariance 붕괴 가설**: ViT의 absolute position embedding 때문에, 동일한 물체가 다른 위치에 있을 때 **모든 패치 값이 달라진다**. Self-attention이 위치 정보를 전 패치에 전파. 따라서 mean-pool 후에도 D-dim 벡터에 위치 잔존.
2. **Information-generation gap 가설**: Vanilla는 representation level에서는 direction 정보를 갖지만, task-specific 답변(letter) 매핑을 학습한 적 없으므로 generation 실패. Ours는 이 매핑을 학습시킨다.

---

# 2. 배경 지식

## 2.1 SigLip ViT 구조

- Input: 384×384 이미지 → 14×14 patch size → **27×27 = 729 patches**
- 각 패치: `patch_embed(patch_i) + position_embed(i)` (= absolute position embedding이 더해짐)
- 이어서 27개 self-attention layer 통과 → `hidden_states[-1]`: (B, 729, 1152)

**Equivariance는 이미 깨져있다**:
- Position embedding은 absolute (patch index별로 학습된 embedding이 더해짐)
- 같은 patch content여도 다른 index에 오면 다른 값
- Self-attention이 이 차이를 전역적으로 전파

## 2.2 LLaVA-Video-7B-Qwen2 Pipeline

```
Frames (8, 3, 384, 384)
   ↓ SigLip ViT (frozen)
Vision tokens (8, 729, 1152)           ← pre-projection
   ↓ mm_projector (mlp2x_gelu)
Projected tokens (8, 729, 3584)         ← post-projection
   ↓ prepare_inputs_labels_for_multimodal
         (vision + text tokens merged)
Input embeddings → Qwen2-7B LLM (28 decoder layers)
   ↓
Last token hidden state (3584-d) at each layer
   ↓ LM head
Next token logit
```

- **mm_projector**: `Linear(1152→3584) → GELU → Linear(3584→3584)` — 각 patch에 독립적으로 적용 (pointwise MLP)
- **Vision encoder (SigLip)는 frozen**; training은 mm_projector + LLM (LoRA)만 건드림

---

# 3. 데이터 및 모델

## 3.1 Dataset: R2R_video_1500

- Location: `/local_datasets/vlm_direction/vlm_direction_testbed/R2R_video_1500/`
- 4 conditions: **shape_color, shape_place, obj_color, obj_place**
- 각 condition: **6000 videos** (4 directions × 1500 each, 균등)
- 32 frames, 384×384, 4 fps
- Random start/end positions, linear motion
- Metadata: `id`, `direction` (up/down/left/right), `speed`, `start_pos`, `obj_class`, `place_class`, …

### 조건 설명

| 조건 | 배경 | 물체 | 복잡도 |
|---|---|---|---|
| `shape_color` | 단색 (color patch) | 도형 (geometric shape) | 최저 |
| `obj_color` | 단색 | 실제 COCO 물체 | 중 (물체 복잡) |
| `shape_place` | 실제 장소 (Places365) | 도형 | 중 (배경 복잡) |
| `obj_place` | 실제 장소 | 실제 물체 | 최고 (real-world simulation) |

## 3.2 MCQ (for LLM hidden state probing)

- Location: `dataset/R2R_4way_1500_json/*.json`
- Question: `"From the viewer's perspective, in which direction is the object moving in this video?"`
- **Candidates: 4 options** (cardinal only: Up, Down, Left, Right)
- Candidates shuffled per video; answer는 "A"–"D" 중 하나
- GT direction은 4-way (up/down/left/right) — candidates와 정확히 일치
ctor 버전의 결과는 `results/.backup_8way/`)

## 3.3 Models

| Model | Description | Checkpoint |
|---|---|---|
| **vanilla** | No fine-tuning | `lmms-lab/LLaVA-Video-7B-Qwen2` |
| **baseline** | LoRA (LLM) + mm_projector train, data = `shape_simple_new`, LM loss only | `4combo_new/work_dirs/..._baseline_shape_simple_new_...` |
| **delta_direct** | Above + explicit **direction regression loss** on delta of pooled vision tokens | `..._delta_direct_shape_simple_new_...` |
| **channel_gate** | Above + channel-wise attention gating module | `..._channel_gate_shape_simple_new_...` |

셋 다 `shape_simple_new` 데이터로 학습되었으므로, 이 분석에서 **obj_place는 distribution shift를 동반한 unseen 조건**이다.

## 3.4 Primary condition

논문 narrative의 중심 조건: **obj_place** (가장 현실적; 모델간 차이가 가장 잘 드러남)

---

# 4. 용어 정리 (Glossary)

## Token / Feature

| 용어 | 의미 |
|---|---|
| **Patch token** | SigLip이 이미지를 14×14로 나눈 각 조각에 대응하는 벡터 (729 개, 1152-d) |
| **Pre-projection token** | SigLip 마지막 hidden state. (B, 729, 1152) |
| **Post-projection token** | mm_projector 통과 후. (B, 729, 3584) |
| **Mean-pooled (over 729)** | `feat.mean(dim=1)` — 프레임 하나당 1 vector. `pre_mean`: (T, 1152), `post_mean`: (T, 3584) |
| **LLM hidden** | Qwen2-7B LLM 각 layer의 **last token 위치**의 hidden state (3584-d) |
| **Frame-level sample** | 한 프레임의 mean-pooled 벡터 하나 = 샘플 1개 |
| **Video-level sample** | 한 영상 전체에서 derive된 벡터 하나 (delta, mid frame, T-mean 등) |

## Derived vectors

| 이름 | 정의 | 용도 |
|---|---|---|
| **Single frame** | `feat[:, t_mid, :]` (중간 프레임 mean-pooled) | Position, Object, Direction sanity |
| **Delta** | `feat[:, -1, :] − feat[:, 0, :]` | Direction (2-frame difference) |
| **T-mean** | `feat.mean(axis=1)` (시간축 평균) | 시간 순서 제거 → direction은 사라지고 semantic 강조 |
| **8-frame stack** | `feat.reshape(N, T*D)` (모든 프레임 펼침) | Full temporal info, LLM-style input에 가까움 |

## Targets

| 이름 | 정의 |
|---|---|
| **Position (x, y)** | Metadata에서 계산된 프레임별 물체 픽셀 좌표 (`start_pos + speed * direction_unit * t`) |
| **Direction** (4-way) | up / down / left / right |
| **GT Letter** | MCQ에서 정답 option의 위치 index (0=A … 3=D). Candidates가 shuffled 되어 있으므로 "direction"과 1:1 매칭이 아님 (chance = 0.25) |
| **Object class** | 26 COCO classes (obj_*) 또는 10 shapes (shape_*) |

## Metrics

| 이름 | 의미 |
|---|---|
| **R² (R-squared)** | Regression이 target 분산의 몇 %를 설명하는가 (1 = 완벽, 0 = chance) |
| **F-statistic (F-stat)** | One-way ANOVA. 한 dim이 class를 얼마나 분리하는지 (between-class / within-class variance) |
| **Top-k dim** | 특정 task에 대한 F-stat 혹은 \|corr\| 상위 k개 dim index |
| **Overlap k1/k2** | 두 top-k dim 집합의 교집합 크기 |

---

# 5. 방법론: Linear Probing

## 왜 Linear Probe인가

- **단순성**: linear model (Ridge / Logistic)만 사용, capacity 편향 제거
- **Interpretability**: linearly decodable = 정보가 "선형으로 읽을 수 있는 형태"로 존재
- **비교 공정성**: 모델간 비교 시 probe model은 동일

## GPU Probe 구현

### Ridge Regression (closed-form)

```python
# w = (X'X + αI)^-1 X'y, bias 제외
X_b = cat([X, ones], dim=1)
reg = α * eye(D+1); reg[-1, -1] = 0   # bias no regularize
w = torch.linalg.solve(X_b.T @ X_b + reg, X_b.T @ y)
pred = X_te_b @ w
R² = 1 − ss_res / ss_tot
```
α = 1.0, 5-fold CV, StandardScaler는 fold별 재학습 (train fit, test transform)

### Classification

```python
model = nn.Linear(D, n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
for epoch in range(300):
    loss = cross_entropy(model(X_train), y_train)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
acc = (model(X_test).argmax(1) == y_test).float().mean()
```
full-batch Adam, 300 epochs, 5-fold StratifiedKFold (class balance 유지)

### F-statistic (per dimension)

```python
for d in range(D):
    groups = [X[y == c, d] for c in range(n_classes)]
    f_stat[d] = f_oneway(*groups).statistic
```
scipy의 one-way ANOVA. NaN은 0으로 대체.

---

# A. Vision tokens carry position information

## A.1 실험 설정 (Experiment A — basic probes)

- **목적**: Q1 — mean-pool 후에도 D-dim 벡터에 위치 정보가 남는가?
- **Data**: `features/vanilla_obj_place.npz`
  - 6000 videos × 8 frames (32 frames 중 uniform sample)
  - → 48,000 frame-level samples
  - Subsample 15,000 for speed
- **Input**:
  - Pre-proj: (T, 1152) per video, mean-pooled over 729 patches
  - Post-proj: (T, 3584)
- **Target**: 해당 프레임의 물체 (x, y) 픽셀 좌표 (metadata에서 계산)
- **Probe**: Ridge regression (α=1.0), 5-fold KFold CV
- **Baseline**: R² ≈ 0 (chance)

### Position 좌표 계산

```python
sample_idx = np.linspace(0, 31, 8, dtype=int)   # 8 uniformly sampled
for meta in metadata:
    sx, sy = meta["start_pos"]
    dx, dy = direction_map[meta["direction"]] * meta["speed"]
    pos[t] = [sx + dx*t, sy + dy*t]    # linear motion model
```

## A.2 결과 (obj_place, N = 6000 videos)

| Metric | Pre-proj (1152-d) | Post-proj (3584-d) | Chance |
|---|---|---|---|
| **Position R² (x)** | **0.806** | **0.839** | 0 |
| **Position R² (y)** | **0.871** | **0.885** | 0 |
| Direction (single frame) | 0.380 | 0.347 | 0.25 |
| **Direction (delta)** | **0.874** | **0.886** | 0.25 |
| Direction (T-mean) | 0.320 | 0.305 | 0.25 |
| Direction (8-frame stack) | 0.867 | 0.779 | 0.25 |
| Object class (26 cls, single) | 0.785 | 0.746 | 0.038 |
| Object class (T-mean) | 0.802 | 0.767 | 0.038 |

## A.3 해석

1. **Q1에 대한 답: YES, 매우 강하게.**
   - 729개 패치를 평균내도 pre-proj R²(x) = 0.81, R²(y) = 0.87. Linear probe로 거의 복원 가능.
   - SigLip의 pos_emb + self-attention propagation이 만드는 효과.

2. **Projector 통과 후 R² 오히려 상승** (pre 0.806 → post 0.839 for x). mm_projector가 위치 정보를 정제하여 LLM에 전달.

3. **Direction**:
   - Single frame (0.38): 당연한 sanity check. 한 프레임으론 "어디에 있는지"만 알지 "어디로 갈지"는 모름.
   - **Delta = 0.87** (chance 0.25): 두 프레임 mean-pool 벡터의 차이만으로 4-way direction 거의 완벽 예측.
   - T-mean (0.32): 시간 순서 제거 → 방향 정보 붕괴. **순서가 direction의 핵심**.
   - Stack (0.87): delta와 비슷. LLM-style full input도 대등한 decodability.

4. **Object class** (26 cls, chance 3.8%): 78%–80% 로 매우 높음 → semantic 정보도 mean-pooled 벡터에 강하게 존재.

---

# B. Position is encoded in a distributed subspace

## B.1 실험 설정 (Experiment B — subspace / null ablation)

- **목적**: Q2 — 위치 정보가 소수 dim에 집중되어 있나, 전체 D에 분산?
- **Data**: 같은 `vanilla_obj_place.npz`, pre-projection만 사용
- **Methods**:
  1. **PCA dimensionality**: PCA(1000 components) 학습 → top-k components만 써서 Ridge probe
  2. **Position-nulled ablation**: per-dim `|corr with (x, y)|`로 ranked top-k 제거 후 나머지 (D-k) dim으로 probe

### PCA detail
```python
pca = PCA(n_components=1000)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X_sub))
for k in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
    R²(x) = cv_r2(X_pca[:, :k], pos_x)
    R²(y) = cv_r2(X_pca[:, :k], pos_y)
    var_explained = pca.explained_variance_ratio_[:k].sum()
```

### Position-correlated dim ranking
```python
corr_x[d] = corrcoef(X[:, d], pos_x)
corr_y[d] = corrcoef(X[:, d], pos_y)
corr_pos[d] = sqrt(corr_x[d]² + corr_y[d]²)
top_pos[k] = argsort(-corr_pos)[:k]
```

### Null ablation
Top-k position dim 제외한 remaining (1152-k) dim으로:
- Object probe (mid frame, N = 6000)
- Direction delta probe (N = 6000)
- Position R² probe (N = 15,000 frames subsample)

## B.2 결과

### B.2.1 PCA dimensionality (obj_place pre-proj)

| PCA k | Variance explained | R²(x) | R²(y) |
|---|---|---|---|
| 1 | 30.4% | 0.0003 | 0.0000 |
| 5 | 45.1% | 0.0025 | 0.0194 |
| 10 | 57.3% | 0.0069 | 0.0195 |
| 20 | 72.4% | 0.0094 | 0.0288 |
| 50 | **91.7%** | **0.0255** | 0.0493 |
| 100 | 95.8% | **0.6998** | **0.7071** |
| 200 | 97.5% | 0.7560 | 0.7978 |
| 500 | 99.1% | 0.7830 | 0.8407 |
| 1000 | 99.9% | 0.8026 | 0.8677 |

### B.2.2 Null ablation (pre-proj)

| 제거한 top-k pos dims | Object acc | Direction (delta) | Position R²(x) |
|---|---|---|---|
| 0 (full) | 0.7843 | 0.8743 | 0.8064 |
| 10 | 0.7870 | 0.8620 | 0.6299 |
| 50 | 0.7870 | 0.8580 | 0.6186 |
| 100 | 0.7858 | 0.8517 | 0.5984 |
| 200 | 0.7842 | 0.8447 | 0.5516 |
| **500** | **0.7788** | **0.7988** | **0.4116** |

## B.3 해석

### 1) Position은 low-variance tail에 분산 인코딩

- PCA k = 50 (variance 92%)로는 R²(x) = 0.03 — **거의 전혀 복원 안 됨**.
- 그런데 **k = 100 (variance 96%)에서 갑자기 0.70**으로 점프.
- 의미: 분산이 큰 상위 PC (variance 중 92%)는 위치와 **무관한 방향**으로 정렬되어 있다. 위치 정보는 51–100번째 PC들에 존재 (variance 단 4%만 추가).

**중요**: PCA = **variance를 최대화하는 방향**. 우리 observation은 vision tokens의 **main variance는 semantic content (e.g. object class)**를 담고 있고, 위치 정보는 그보다 작은 2차 축들에 숨어있다는 것.

### 2) Null ablation: top-500 제거해도 정보 잔존

- Top-500 pos-correlated dim 제거 → 나머지 652 dim에서도 R²(x) = 0.41
- 즉 위치 정보가 **전체 1152 dim에 퍼져있음** (top-500이 전부가 아님)
- **Object/Direction은 거의 영향 없음** (object: 0.78→0.78, dir_delta: 0.87→0.80)
  - Object acc는 top-500 제거 후에도 유지 → **semantic dim 집합과 position dim 집합이 분리**
  - Direction은 delta 기반이므로 position dim에 살짝 의존 → top-500 제거 시 0.87→0.80으로 소폭 하락

### 3) Q2에 대한 답

**위치 정보는 low-rank subspace가 아니라 distributed representation.** 단 수십 개 dim을 제거하는 것으로는 제거 불가. 이는 method design 관점에서 중요:
- "위치 담당 dim만 찾아서 amplify" 같은 specific한 조작은 어려움
- **전체 D에 분산된 신호를 동시에 조작하는 projector-level tuning**이 필요 (→ ours가 그렇게 함, F 섹션)

## B.4 (추가) Complexity에 따른 subspace 구조 변화 — 4 condition 비교

**핵심 질문**: obj_place 결과 ("position은 low-variance tail에 분산")는 **시각적 복잡도가 높아서** 나타난 현상인가, 아니면 SigLip이 본질적으로 그런가? → shape_color (가장 단순) ~ obj_place (가장 복잡) gradient로 답한다.

### 실험 설정
- 4 conditions 모두 동일 subspace analysis (N=6000 each, pre-proj, 15,000 frame subsample)
- Reported: PCA k별 R²(x), variance explained, null ablation

### B.4.1 PCA R²(x): position이 PC 어디쯤에 있나

| PCA k | shape_color | shape_place | obj_color | **obj_place** |
|---|---|---|---|---|
| 1 | 0.005 | ~0 | ~0 | ~0 |
| 10 | 0.047 | ~0 | 0.009 | 0.007 |
| 20 | 0.108 | 0.002 | 0.027 | 0.009 |
| **50** | **0.859** ★ | 0.006 | 0.077 | 0.026 |
| 100 | 0.933 | 0.734 | 0.273 | 0.700 |
| 500 | 0.980 | 0.817 | 0.693 | 0.783 |
| 1000 | 0.984 | 0.854 | 0.760 | 0.803 |

**Variance explained at same k**:

| PCA k | shape_color var | shape_place var | obj_color var | obj_place var |
|---|---|---|---|---|
| 1 | 0.219 | 0.213 | **0.088** | **0.304** |
| 50 | 0.926 | 0.978 | 0.645 | 0.917 |

### B.4.2 Null ablation: top-k position dim 제거 후 R²(x)

| Remove top-k | shape_color | shape_place | obj_color | obj_place |
|---|---|---|---|---|
| 0 | 0.985 | 0.860 | 0.773 | 0.806 |
| 50 | 0.976 | 0.798 | 0.739 | 0.619 |
| 200 | 0.972 | 0.765 | 0.698 | 0.552 |
| **500** | **0.956** | 0.665 | 0.565 | **0.412** |
| Δ (0 → 500) | **−0.03** | −0.20 | −0.21 | **−0.39** |

### B.4.3 해석

**시각적 복잡도와 position subspace 구조는 강하게 결합**:

1. **shape_color (가장 단순)**: Position이 **main variance**의 일부 → PCA top-50 (var 93%)만으로 R²=0.86. 또한 **모든 dim이 같은 position을 redundantly 인코딩** → top-500 제거해도 0.96 잔존.

2. **obj_place (가장 복잡)**: Position이 **low-variance tail**에 숨어있음 → PCA top-50에서 R²=0.03, k=100 이후에야 급점프. 그리고 dim별로 **기여 차이 큼** → top-500 제거하면 R² 급락 (−0.39).

3. **obj_color vs shape_place** 교차 비교로 재확인: 
   - shape_place (실배경+도형): k=100에서 0.73으로 빠르게 회복 (도형이 단순해서 배경만이 variance source)
   - obj_color (단색+실물체): k=100에서 0.27밖에 안 됨 — 실물체의 visual variance가 매우 커서 position을 더 깊게 묻어버림

4. **"가장 dominant한 PC1이 position과 얼마나 unrelated한가"**:
   - shape_color: PC1 variance 22%, R²(x) 0.005 (약간 연관)
   - **obj_place: PC1 variance 30%, R²(x) ~0 (완전 무관)** — 한 방향에 가장 큰 variance가 쏠려있지만 위치와 전혀 관계 없음 (real-world texture/clutter로 추정)

### B.4.4 Method design 시사점

1. **Simple scene에서만 "PCA/분산 큰 축 = position"**. Real-world (obj_place)에서는 정반대. 따라서 **PCA 기반 dim selection으로 position을 찾는 휴리스틱은 복잡 scene에서 실패**.
2. **복잡 scene일수록 position signal이 약하고 dim별 차이 큼** → 전역 개입이 더 필요하고, explicit direction supervision (delta_direct 계열)이 이득을 보는 이유.
3. Analysis section F에서 본 **obj_place에서 ours의 F-stat 상승이 shape_simple 기준 training보다 작은 것**도 이 현상의 연장 — 복잡 scene에서는 projector만으론 부족, LLM 내부 supervision (plan의 `llm_delta_multilayer`)이 필요한 이유.

---

# C. Direction emerges from temporal dynamics

## C.1 실험 설정 (Experiment C — temporal probes)

- **목적**: Q3 — 방향 정보가 vision token에 어떻게 존재하며, 얼마나 decodable한가?
- **Data**: `vanilla_obj_place.npz`, N = 6000 videos × 8 frames
- **4가지 derived vector** 비교:

| Vector | 수식 | Sample 단위 | 시간 정보 |
|---|---|---|---|
| Single frame | `feat[:, 4, :]` | video | 없음 |
| T-mean | `feat.mean(axis=1)` | video | 시간 순서 제거, 평균 위치만 |
| Delta | `feat[:, -1, :] − feat[:, 0, :]` | video | 위치 변화량 |
| 8-frame stack | `feat.reshape(N, 8*D)` | video | 전체 (LLM-style) |

- **Probe**: 4-way (up/down/left/right) LogisticRegression
- 8-frame stack은 8×D feature라 overfitting 위험; 그래서 N=6000으로 sufficient

## C.2 결과 (pre-proj; obj_place)

| 방법 | Direction acc | Chance |
|---|---|---|
| Single frame (mid) | **0.380** | 0.25 |
| T-mean | **0.320** | 0.25 |
| **Delta** | **0.874** | 0.25 |
| 8-frame stack | 0.867 | 0.25 |

## C.3 해석

### 1) Single frame 38% — sanity check
- 단일 프레임은 이론적으로 방향 모름
- chance 위의 38%는 N=6000에서 linear model이 잡는 약한 statistical bias (예: "mid frame에서 x가 크면 right 방향일 조건부 확률 약간 높음")

### 2) T-mean 32% — chance 근처 → 시간 순서가 direction의 핵심
- 8 frames의 mean은 궤적 중심점 → 위치 정보는 남지만 **방향 정보는 붕괴**
- 이는 "direction = position의 temporal derivative"를 확인

### 3) Delta = 87% — 핵심 결과
- **Mean-pooled 벡터 2개의 차이만으로 4-way direction 거의 완벽 예측**
- 왜 작동하나?
  - Section A에서 확인: pre_mean_t ≈ W · pos_t + (기타 invariant 정보)
  - delta = pre_mean_T − pre_mean_0 ≈ W · (pos_T − pos_0) = W · (velocity vector × T)
  - Direction은 velocity vector의 부호 → delta에 그대로 실림

### 4) Stack = 87% — LLM이 보는 것과 같은 포맷으로도 decodable
- Delta와 비슷한 성능 → Section H에서 LLM last token probe가 유사한 수준 달성하는 근거

---

# D. Position/Direction vs Semantic — dim-level separation

## D.1 실험 설정 (Experiment D — dim overlap)

- **목적**: Q4 — 같은 D-dim 벡터 내에서 position/direction/semantic이 같은 dim을 공유? 분리?
- **Data**: `vanilla_obj_place.npz`, pre/post-proj 둘 다
- **Top-k 기준 (k = 50)**:

| Target | Ranking method |
|---|---|
| Position | per-dim `sqrt(corr_x² + corr_y²)` (frame-level, 15,000 samples subsample) |
| Direction | per-dim ANOVA F-stat across 4 groups, on **delta** vector (6000 videos) |
| Object class | per-dim ANOVA F-stat across 26 groups, on **mid frame** (6000 videos) |

- **Overlap metric**: `|top_A ∩ top_B|` out of 50
- **Random expected**: `50²/D` (D = 1152 for pre, 3584 for post)

## D.2 결과

### Top-50 overlap (obj_place)

| Pair | Pre-proj (D=1152, random ≈ 2.2) | Post-proj (D=3584, random ≈ 0.7) |
|---|---|---|
| **Position ∩ Direction** | 17/50 | 17/50 |
| **Position ∩ Object** | **1/50** | **0/50** |
| **Direction ∩ Object** | **0/50** | **1/50** |

## D.3 해석

### 1) Position ≈ Direction (어느 정도 공유)
- 17/50 overlap (random 2.2 대비 8배) — 위치를 담는 dim의 일부가 direction F-stat도 높음
- **직관**: direction은 position의 시간 변화이므로, position-encoding dim의 delta가 direction 신호가 됨

### 2) Position vs Object 완전 분리
- **1/50 (pre), 0/50 (post)** — random 수준 혹은 그 이하
- 위치 담당 dim과 semantic 담당 dim이 **완전히 다른 부분공간**에 존재
- **Projector 통과 후 더 깨끗하게 분리** (1→0)

### 3) Q4에 대한 답

**Spatial(position/direction)과 Semantic(object)은 dim-level로 분리된 subspace를 사용**. 이는:

- Method design 관점에서 중요: spatial dim만 amplify해도 semantic을 손상시키지 않음 (우리 ours가 바로 이 특성을 활용)
- 근데 "top-50 dim이 정말 independent한가?"는 더 엄밀한 질문. Cross-probe 실험 (각 top-50 dim으로 다른 task 예측)에서는 cross-accuracy가 의외로 높게 나오는데, 이는 **주된 신호는 분리되지만 약한 secondary signal은 곳곳에 존재**함을 의미.

---

# E. Visual complexity degrades direction encoding

## E.1 실험 설정 (Experiment E — complexity gradient)

- **목적**: Q5 — 배경/물체 복잡도가 direction encoding을 얼마나 약화?
- **Data**: 4 conditions (shape_color, shape_place, obj_color, obj_place), **각 1500 videos subsample** (balanced: 375 per direction)
  - 전체 6000씩 써서 비교하면 F-stat이 N에 의존하므로, **같은 샘플 수로 통일**
- **Features**: pre-projection (mean-pooled)
- **Probes**:
  - Position R²(x, y): frame-level (1500 × 8 = 12,000 samples)
  - Direction delta: video-level (N = 1500)
  - F-stat per dim on delta: direction-discriminative dim 수 측정

## E.2 결과

| 조건 | R²(x) | R²(y) | delta dir | F_top50 mean | F > 5 dims | F > 10 dims |
|---|---|---|---|---|---|---|
| **shape_color** | **0.986** | **0.995** | **0.992** | **309.4** | 1084 | 993 |
| `obj_color` | 0.848 | 0.949 | 0.961 | 109.6 | 982 | 747 |
| `shape_place` | 0.883 | 0.932 | 0.865 | 77.6 | 722 | 447 |
| **`obj_place`** | 0.852 | 0.904 | **0.783** | **69.4** | **610** | 334 |

## E.3 해석

### 1) Direction 성능의 drastic한 하락
- shape_color: 99.2% → obj_place: **78.3%** (−21%p)
- 배경 + 물체 복잡도가 **동시에** 작용할 때 가장 심각

### 2) 배경이 물체보다 더 큰 영향
- 배경만 추가 (shape_color → shape_place): 99.2 → 86.5 (−12.7%p)
- 물체만 복잡화 (shape_color → obj_color): 99.2 → 96.1 (−3.1%p)
- **배경 complexity가 direction encoding의 주적**

### 3) Direction-strong dim 수의 감소
- F > 5 기준: 1084 → 610 (43% 감소)
- F > 10: 993 → 334 (66% 감소)
- **복잡한 scene에서는 "direction을 강하게 구별하는 dim"이 급감** → linear probe가 쓸 수 있는 feature pool이 좁아짐

### 4) Q5에 대한 답과 method 시사점

**YES, 복잡도(특히 배경)가 direction encoding을 약화시킨다.** 이는 우리 ours method의 motivation:
- Real-world에서는 배경 noise가 direction signal을 희석
- Direction-focused mechanism (delta_direct, channel_gate)이 이 희석을 보상할 수 있음
- Section F–H에서 실제로 이 보상이 일어나는지 검증

---

# F. Vanilla → Baseline → Ours — projector-level comparison

## F.1 실험 설정 (Experiment F — trained model comparison)

- **목적**: Q6 — 학습된 모델은 무엇이 어떻게 바뀌었나?
- **Data**: `features/multiproj_obj_place.npz` (N = 6000 videos)
  - 각 영상에 대해 **4개의 projector** (vanilla + baseline + delta_direct + channel_gate) 를 per-patch 적용 후 mean-pool
  - 같은 SigLip output에 다른 projector를 태우므로 **공정 비교**
- **Probes (post-projection)**:
  - Position R²(x, y) on subsampled 15,000 frames
  - Direction delta on N = 6000
  - Object (26 cls) on mid frame
  - Direction F-stat per dim, top-50 mean
- **Vanilla 기준 분석**:
  - `overlap_w_vanilla`: `|trained_top50 ∩ vanilla_top50|`
  - `vanilla_top50_strengthened`: vanilla의 top-50 direction dim 중 trained에서 F-stat이 올라간 개수
  - `F_on_vanilla_top50`: vanilla top-50의 평균 F-stat이 trained에서 얼마나 변했나

## F.2 결과 (obj_place, N = 6000 videos, post-proj)

### 메인 테이블

| Model | R²(x) | R²(y) | delta_dir | obj (26 cls) | F_dir mean | F_dir top50 mean | overlap w/ vanilla | v_top50 strengthened | F on v_top50 |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 0.8392 | 0.8853 | 0.8868 | 0.7458 | — | **443.4** | — | — | — |
| baseline | 0.8397 | 0.8861 | 0.8880 | 0.7447 | — | 449.5 | **47/50** | 38/50 | 449 |
| **delta_direct** | **0.8524** | **0.8964** | **0.9072** | 0.7448 | — | 451.8 | 34/50 | 19/50 | 419 |
| **channel_gate** | **0.8539** | 0.8949 | **0.9042** | 0.7462 | — | **463.7** | 36/50 | 25/50 | 439 |

*R² subsampled 15,000 frames; delta_dir은 video-level N=6000; obj은 mid frame N=6000*

## F.3 해석

### 1) Accuracy 변화 (vs vanilla)

| Metric | vanilla | baseline | delta_direct | channel_gate |
|---|---|---|---|---|
| Position R²(x) | 0.839 | ≈ | **+0.013** | **+0.015** |
| delta_dir | 0.887 | ≈ | **+0.020** | +0.017 |
| obj acc | 0.746 | ≈ | ≈ | ≈ |

- **Baseline: vanilla와 거의 동일** (모든 metric 차이 < 0.005)
- **Ours: direction/position ↑, object 유지** — semantic을 건드리지 않고 spatial에만 개입

### 2) Dim 구조 변화

**Baseline**: overlap 47/50 → vanilla와 거의 같은 dim 사용
- "학습이 거의 안 된" 수준으로 dim 구조 보존
- 다만 38/50 strengthened — 같은 dim에서 F-stat 살짝 상승

**Delta_direct / Channel_gate**: overlap 34–36/50 → 15개 가량 다른 dim
- vanilla의 top-50 중 일부를 "버리고" 다른 dim을 top-50에 새로 세움
- `F_on_vanilla_top50` 값: delta_direct 419 (↓ 24 from 443), channel_gate 439 (≈)
- 즉 **vanilla가 쓰던 top-50 direction dim이 더 강해진 게 아니라, 새로운 dim이 더 강해짐**

### 3) F-stat top-50 총 강도

- vanilla: 443
- baseline: 450 (+1.5%)
- delta_direct: 452 (+2%)
- **channel_gate: 464 (+4.6%)**

**obj_place에서 ours의 F-stat 상승은 상대적으로 작음** (이전 `shape_simple_new`에서 측정할 때는 +40%였음). 이유:
- Ours는 `shape_simple_new`로 학습 → obj_place는 distribution shift
- 복잡한 scene에서 direction 신호 자체가 약해 (Section E), training의 효과가 희석됨
- 그럼에도 **direction probe acc는 +2%p** 상승 — dim-level 변화가 작아도 decoder는 활용 가능

### 4) Q6 1차 답변 (Section F만으로)

**Baseline은 representation 수준에서 사실상 vanilla. Ours는 약간의 dim 재배치로 direction에 약간 유리한 representation을 만든다.** 다만 이 차이만으로는 baseline과 ours의 **generation 성능 격차를 충분히 설명하지 못함**. → Section H에서 LLM propagation을 봐야 진짜 이유가 드러남.

---

# G. Projector weight change analysis

## G.1 실험 설정

- **목적**: Q6 — Baseline과 ours가 **실제로 projector를 얼마나 바꿨는지** 직접 weight 비교
- **대상**: `mm_projector.0.weight`, `mm_projector.2.weight` (두 Linear layer)
- **Metrics**:
  - `rel_diff`: `||W_trained − W_vanilla||_F / ||W_vanilla||_F`
  - `row_cos_min`: 각 output dim을 만드는 weight row 벡터간 cosine similarity의 minimum
  - `row_cos_min`이 1에 가까우면 weight의 "방향"은 보존된 것

### Weight 로딩
```python
# Vanilla projector
for shard in model.safetensors.index["mm_projector"]:
    load vanilla weights

# Trained projector
state = torch.load(f"{trained_dir}/non_lora_trainables.bin")
trained_proj = filter "mm_projector.*" from state
```

## G.2 결과

| Model | Layer 0 (Linear 1152→3584) | | Layer 2 (Linear 3584→3584) | |
|---|---|---|---|---|
|  | `rel_diff` | `row_cos_min` | `rel_diff` | `row_cos_min` |
| **baseline** | **1.3%** | 0.9992 | 1.6% | 0.9991 |
| **delta_direct** | **4.7%** | 0.9864 | 5.8% | 0.9923 |
| **channel_gate** | **4.8%** | 0.9895 | 5.9% | 0.9919 |

## G.3 해석

### 1) Baseline: projector 거의 안 변함
- 1–2% 미세 변화. 사실상 vanilla projector와 동일
- LoRA + LM loss로는 **projector update가 거의 없음** — downstream task에 맞게 projector가 튜닝되지 않음
- F 섹션에서 본 "representation ≈ vanilla" 현상의 근거

### 2) Ours: 5–6% 변화, 방향은 보존
- `row_cos_min > 0.98` → 각 output dim을 만드는 linear combination의 **방향**은 거의 그대로
- 변한 건 주로 **norm / bias** — 즉 각 dim의 "크기 / shift" 수준의 미세 조정
- 해석: vanilla의 dim 의미를 유지하면서, 특정 dim들을 직교 방향으로 slight하게 push

### 3) 왜 작은 변화가 큰 효과를 내나

`F-stat = between-class variance / within-class variance`. Weight cosine ≈ 1이므로 각 dim이 인코딩하는 "방향"은 그대로 → **class별 centroid가 미세하게 이동**하면 between-class variance가 커지고 F-stat이 상승. 작은 weight shift가 "보다 선명하게 class를 분리"시키는 효과.

### 4) 왜 ours의 효과가 obj_place에서는 약한가 (F 섹션과 연결)

- 학습 데이터는 `shape_simple_new` (단순) → weight는 단순 조건에 특화되어 밀림
- obj_place는 실물체·실배경이라 이 밀림이 그대로 적용되기 어려움
- 그래도 5% 수준의 변화는 남아서 direction probe acc 소폭 상승

---

# H. LLM hidden state propagation — the information–generation gap

## H.1 실험 설정 (Experiment H — LLM probe)

이 섹션이 우리 분석의 **하이라이트**. Vision token representation을 보는 것만으로는 "왜 vanilla는 answer letter를 못 내는가"를 설명 못 함. LLM 내부에서 vision info가 어떻게 변환되는지 봐야 함.

### Prompt

R2R_4way_1500_json의 MCQ 포맷 그대로 사용:

```
<image>
Question: From the viewer's perspective, in which direction is the object moving in this video?
Options:
(A) Down
(B) Left
(C) Up
(D) Right
Answer with the letter of the correct option only.
```

- 4-way MCQ (UDLR, distractor 없음 — diagonals 제거)
- Candidates shuffled per video → letter↔direction은 random mapping (chance = 0.25)
- **Critical**: `tokenizer_image_token`으로 `<image>`를 `IMAGE_TOKEN_INDEX` 로 치환해야 `prepare_inputs_labels_for_multimodal`이 vision token을 올바르게 병합함

### Forward pass

```python
with torch.no_grad():
    (_, _, attn, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None,
        images=[pv], modalities=["video"])
    outputs = model.forward(
        input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attn,
        output_hidden_states=True, return_dict=True)
hs = outputs.hidden_states   # tuple of (n_layers+1), each (1, seq_len, 3584)
```

- `hidden_states[0]` = embedding output (before any transformer block)
- `hidden_states[k]` for k ∈ [1, 28] = after transformer block k−1
- **Probe target**: `hs[k][0, -1, :]` = last token hidden at layer k

### Layers probed
`[0, 1, 4, 8, 12, 16, 20, 24, 28]` — 얕은/중간/깊은 layer 포괄

### Data subsample
obj_place.json의 6000 MCQ items 중 **방향별 500 balanced sampling = 2000 items** (seed=42)

### Model loading

- **Vanilla**: `LlavaQwenForCausalLM.from_pretrained(MODEL_BASE, torch_dtype=fp16)`
- **Trained**:
  1. Load vanilla base
  2. Load `non_lora_trainables.bin` (projector updates + misc. trainable) → `load_state_dict(strict=False)`
  3. **Manually merge LoRA** (PEFT의 bitsandbytes import 이슈 우회):
     ```python
     # From adapter_model.safetensors:
     # base_model.model.<path>.lora_A.weight  (r × in)
     # base_model.model.<path>.lora_B.weight  (out × r)
     # Merge: W_new = W_old + (α/r) * B @ A
     ```
  - α/r = 128/64 = 2.0
  - 196 LoRA layers merged per model

### Parallel extraction
- 4 GPU × 1 model each (gpu0 = vanilla, gpu1 = baseline, gpu2 = delta_direct, gpu3 = channel_gate)
- 2000 videos × ~0.3 s/video per GPU → ~10 min per model, all in parallel

### Probes (on 3584-d last-token hidden)
- **Direction** (4-way: up/down/left/right) — chance = 0.25
- **GT Letter** (4-way: 0=A … 3=D, since options are shuffled 4-way) — chance = 0.25

Object probing omitted for this section (focus on direction↔answer pipeline).

## H.2 결과 (obj_place, N = 2000) — 4-way MCQ

### Direction probe (4-way, chance = 0.25)

| Layer | Vanilla | Baseline | Delta_direct | Channel_gate |
|---|---|---|---|---|
| 0 (embed) | 0.250 | 0.250 | 0.250 | 0.250 |
| 1 | 0.540 | 0.544 | 0.605 | **0.612** |
| 4 | 0.735 | 0.738 | 0.774 | **0.790** |
| 8 | 0.696 | 0.755 | 0.782 | **0.799** |
| 12 | 0.725 | 0.824 | **0.826** | 0.809 |
| 16 | 0.689 | **0.846** | 0.830 | 0.834 |
| 20 | 0.665 | 0.870 | 0.874 | **0.880** |
| 24 | 0.612 | 0.874 | 0.871 | **0.886** |
| **28 (last)** | **0.577** | 0.856 | 0.860 | **0.867** |

### GT Letter probe (4-way, chance = 0.25)

| Layer | Vanilla | Baseline | Delta_direct | Channel_gate |
|---|---|---|---|---|
| 0 (embed) | 0.263 | 0.263 | 0.263 | 0.263 |
| 1 | 0.242 | 0.238 | 0.243 | 0.233 |
| 4 | 0.241 | 0.252 | 0.259 | 0.251 |
| 8 | 0.254 | 0.249 | 0.246 | 0.249 |
| 12 | 0.249 | 0.244 | 0.242 | 0.249 |
| 16 | 0.252 | 0.277 | 0.285 | **0.297** |
| 20 | 0.300 | 0.504 | **0.531** | 0.508 |
| 24 | 0.280 | 0.567 | **0.609** | 0.561 |
| **28 (last)** | **0.302** | 0.649 | **0.683** | 0.663 |

> 8-way 분석 결과는 `results/.backup_8way/llm_probe_obj_place.json`에 보존.

## H.3 해석 — 이 분석의 하이라이트

### 1) Layer 0 (embedding): 모두 chance
- 2000 videos 전부에 대해 prompt가 동일 (candidates가 shuffled된 것 외에는)
- Last token은 아직 self-attention이 안 일어났으므로 vision info 흡수 안 됨

### 2) Layer 1–4: Direction 정보 급속 흡수
- Vanilla: chance → 0.54 → 0.74 (L4 peak)
- **Ours가 얕은 layer에서 이미 우위**: L4에서 vanilla 0.74, channel_gate 0.79 (+5%p)
- Fine-tuning이 얕은 attention에서부터 direction info를 더 효율적으로 aggregate

### 3) Layer 8–28: 핵심 현상
| 관점 | Vanilla | Baseline | Delta_direct | Channel_gate |
|---|---|---|---|---|
| **Direction L28** | 0.577 (↓ from peak) | 0.856 | 0.860 | 0.867 |
| **Direction 깊은 layer 유지** | 급격 감쇠 | 강 보존 | 강 보존 | 강 보존 |
| **GT Letter L28** | 0.302 (≈ chance) | 0.649 | **0.683** | 0.663 |
| **GT Letter L20 (early)** | 0.300 | 0.504 | **0.531** | 0.508 |

#### 3a) Direction은 3 모델 모두 깊게 유지됨 (Baseline ≈ Ours)
- Baseline도 L12 이후 direction probe 0.82–0.87로 vanilla (0.58)보다 훨씬 높음
- 즉 **LoRA가 LLM의 attention을 vision token에 더 오래 attend하게** 만드는 효과
- Projector weight는 baseline이 거의 안 변했어도 (Section G), **LLM LoRA adapter가 깊은 layer 정보 유지를 바꾼다**

#### 3b) GT Letter는 L20부터 emerge하고 L28에서 peak — vanilla는 끝까지 chance
- L0–L16: 모든 모델이 chance 0.25 근방 (≤ 0.30) — vision attention만 진행 중, letter mapping은 아직
- **L20에서 trained 3개가 0.50+으로 점프** (vanilla 여전히 0.30)
- L24: trained 0.56–0.61, L28: trained **0.65–0.68**
- **Vanilla는 L28에서도 0.302로 chance 근방** ← 이것이 vanilla가 MCQ를 못 푸는 이유

> **8-way (이전) vs 4-way (현재)의 차이**: 8-way 때는 letter probe 점프가 L28에서만 보였으나, 4-way에서는 **L20부터** 명확. 이유는 distractor가 사라져 mapping 부담이 줄어든 것 — 그러나 **vanilla는 그 mapping을 학습하지 않았으므로 4-way에서도 chance**. Information–generation gap은 더 깨끗하게 드러남.

### 4) Information–Generation Gap (4-way 기준 더 선명)

**Vanilla**:
- Direction probe L4: 0.74 (direction 정보 강하게 보유)
- Letter probe L28: **0.302** (chance 0.25 거의 그대로 — 정답 letter 출력 못 함)
- **즉 "방향은 알지만 answer letter로 변환을 못 함"** ← 4-way에서 더 깨끗

**Trained**:
- Direction probe L28: 0.86–0.87 (direction 정보 강 보존)
- Letter probe L28: **0.65–0.68** (letter 출력 능력 획득, chance 대비 +40%p+)
- Training의 진짜 효과 = **representation → generation format mapping** 학습

### 5) Delta_direct가 letter mapping 최강 (0.683)

- baseline 0.649 → delta_direct 0.683 (+3.4%p)
- **Delta direction regression loss**가 LLM에 "direction 정보를 answer token 형태로 출력하는 방법"을 가장 잘 가르침
- Channel_gate (0.663)도 baseline보다 약간 우위 — 8-way에서는 baseline과 동급이었으나 4-way에서는 미세 우위
- Direction 정보 자체는 channel_gate가 강함 (L20 0.880, L24 0.886), letter mapping은 delta_direct가 가장 강함 — **explicit direction supervision이 letter head 학습에 더 직접적이라는 결론은 유지**

### 6) Q7–Q9 답

- **Q7**: LLM은 layer 1부터 vision 정보 흡수, peak는 vanilla = L4, trained = L20–24 (layer 위치 뒤로 이동)
- **Q8**: Vanilla vs trained 격차는 L12 이후 direction probe에서, L20 이후 letter probe에서 각각 드러남
- **Q9**: **직접 NO** — direction 정보를 가진다고 letter를 맞추는 건 아니다. Letter mapping은 explicit training (LoRA + 특히 direction loss)로만 얻어짐. 4-way distractor가 사라진 더 쉬운 task에서도 vanilla는 여전히 chance에 머무름 → **gap의 본질은 옵션 수가 아니라 mapping 학습 유무**.

---

# I. Trajectory straightness in representation space

## I.1 실험 설정

비디오 1개당 frame-level mean-pooled 벡터 sequence `x_0, x_1, …, x_{T−1}` (T=8). "직선성(straightness)"을 세 가지 metric으로 측정:

| 지표 | 정의 | 해석 |
|---|---|---|
| **Curvature (°)** | 연속 velocity `v_t = x_{t+1}−x_t`의 consecutive angle `arccos(cos(v_t,v_{t+1}))`의 평균 | 0° = 완전 직선, 90° = 랜덤, >90° = 반대방향 swing |
| **PC1 ratio** | SVD 후 first singular value² / total variance | 1.0 = rank-1 line, 1/(T−1)≈0.14 = 완전 랜덤 |
| **Linear-fit R²** | `X_t = a·t + b` OLS 피팅의 R² | 1.0 = linear temporal evolution |

2종류 분석:
1. **Per-video**: 각 비디오 단독 (노이즈 우세)
2. **Class-mean**: 방향별 (up/down/left/right, 1500 videos each) 평균 trajectory를 계산한 후 straightness 측정 (노이즈 소거)

Dim 레벨: full D (3584 post-proj, 1152 pre-proj), top-K F-stat dims (K ∈ {50, 200}) — baseline F_dir 기준.

## I.2 결과 (obj_place, N = 6000)

### Per-video (노이즈-dominated regime)

| Dim subset | Vanilla | Baseline | Delta_direct | Channel_gate |
|---|---|---|---|---|
| full_D curvature (°) | 103.96 | 103.95 | 103.97 | 103.92 |
| top50 curvature (°) | 103.54 | 103.54 | 103.62 | 103.54 |
| full_D PC1 | 0.538 | 0.538 | 0.537 | 0.538 |
| top50 PC1 | 0.595 | 0.594 | 0.592 | 0.596 |
| full_D R²_line | 0.433 | 0.434 | 0.434 | 0.435 |

→ **4 모델 사실상 동일**. 3584-d fp16 공간 + T=8 프레임이면 개별 프레임 노이즈가 signal을 덮음 (velocity cosine이 평균 0에 가까워 ~104°). Pre-projector (1152-d)도 동일 수준 (104.15°).

### Class-mean (노이즈 소거 후 signal-dominated regime)

| Dim subset | Vanilla | Baseline | Delta_direct | Channel_gate |
|---|---|---|---|---|
| **full_D curvature (°)** | 37.79 | 37.63 | 37.30 | **36.51** |
| **full_D PC1** | 0.833 | 0.832 | 0.826 | 0.819 |
| **full_D R²_line** | 0.686 | 0.689 | 0.680 | **0.693** |
| top50 curvature (°) | 31.44 | 31.22 | 31.54 | **31.01** |
| top50 PC1 | 0.875 | 0.872 | 0.875 | 0.871 |
| top50 R²_line | 0.732 | **0.737** | 0.729 | 0.727 |
| top200 curvature (°) | 32.51 | 32.36 | 32.66 | **31.99** |

Pre-proj reference (SigLip 1152-d, class-mean full_D): curvature = 41.68°, PC1 = 0.832, R² = 0.665.

## I.3 해석

### 1) 노이즈 vs 시그널 레짐
- Per-video curvature ~104° → velocity가 프레임간 거의 무관 (노이즈)
- Class-mean curvature ~37° → 방향별 평균은 분명 직선에 가까움 (PC1 0.83, R² 0.69)
- **직선성 차이는 class-mean에서만 드러남** (per-video로는 model 차이가 signal 아래 묻힘)

### 2) Projector가 직선성을 "약간" 개선
- SigLip pre-proj (class-mean): 41.68°
- Vanilla mm_projector: 37.79° (−3.9°)
- 즉 **base mm_projector 자체가 이미 SigLip 출력을 ~4° 더 straight 하게 만듬** (projection이 direction-relevant subspace로 변환하는 효과)

### 3) Ours는 추가 직선화가 미미 (+0.5–1.3°)
- Vanilla 37.79° → channel_gate **36.51°** (−1.28°)
- Delta_direct 37.30° (−0.49°), baseline 37.63° (−0.16°)
- **Channel_gate가 가장 straight** — 하지만 ±1° 수준이라 dramatic effect 아님
- Top-50 dim에서도 비슷한 순위 유지 (channel_gate 31.01° < delta 31.54° < baseline 31.22° < vanilla 31.44°, 차이 ~0.5°)

### 4) 결론: "Representation straightening"은 우리 method의 주효과가 아님
- Section H에서 본 letter-mapping 격차 (vanilla 0.30 vs ours 0.68) 대비, trajectory geometry는 거의 변하지 않음
- **Direction signal이 이미 vanilla의 mm_projector output에서도 approximately-linear manifold에 있음**
- Fine-tuning의 효과는 **trajectory 모양 바꾸기보다 LLM이 그 trajectory를 letter token으로 읽는 법 학습**
- 즉 (projector 출력 공간에서는) vanilla와 ours가 **geometrically similar**, 차이는 downstream LLM의 해독 능력

### 5) Q 보완
- Q_new: "Ours는 vision token trajectory를 직선화하는가?" → **아니오, 미세한 개선(~1°)만 있음.** 대신 Section H처럼 **LLM-side mapping**에서 차이.
- Q_new 2: "그럼 direction signal의 기원은?" → Pre-proj (SigLip)부터 이미 class-mean trajectory가 linear-like. Projector가 살짝 정제. Fine-tuning은 geometry 거의 유지.

---

# 7. Synthesis: vanilla → baseline → ours 논리

## 7.1 Vanilla: "정보는 있는데 답변을 못 함"

| 관점 | Observation | Section |
|---|---|---|
| Vision level | Position R² = 0.81, delta direction = 0.87 | A, C |
| Dim level | Position 분산 인코딩, semantic과 분리 | B, D |
| Complexity | 배경 복잡도로 direction 약화 | E |
| Projector | 학습 없음 (base weight) | G |
| LLM propagation | Layer 1부터 흡수, L4 peak, 이후 감쇠 | H |
| **Generation** | **Letter probe L28 = 0.302 (≈ chance 0.25)** | H |

→ **Representation 수준에서는 direction 정보가 있다. 문제는 이를 MCQ answer letter로 출력하는 법을 학습하지 않음.**

## 7.2 Baseline: "Projector는 거의 안 바꾸고, LLM LoRA로 direction 보존력 얻음"

| 관점 | Observation | Section |
|---|---|---|
| Projector weight | rel_diff 1.3–1.6%, vanilla와 거의 동일 | G |
| Post-proj dim 구조 | overlap 47/50 with vanilla | F |
| Vision-level probe acc | vanilla와 동일 | F |
| LLM deep-layer direction | **L28 probe 0.86** (vanilla 0.58) | H |
| **Generation** | **Letter L28 = 0.649** (+35%p over vanilla) | H |

→ **LoRA는 projector를 거의 안 건드리지만 LLM attention을 조정해 "vision info를 deep layer까지 유지 + letter로 출력" 능력을 준다.** 단순 LM loss만으로도 상당한 향상 가능.

## 7.3 Ours (delta_direct / channel_gate): "Projector tuning + direction-specific supervision이 추가"

| 관점 | Observation | vs Baseline |
|---|---|---|
| Projector weight | rel_diff 4.7–5.9% (row cos > 0.98) | +3–4%p 더 변함 |
| Post-proj F_dir top50 | 452–464 | +2–14 points |
| Position R²(x), delta_dir | 0.85 / 0.91 (both +0.01–0.02) | 약간 상승 |
| LLM direction at peak | L24 = 0.886 (channel_gate) vs 0.874 (baseline) | 작은 상승 |
| **GT letter L28 (delta_direct)** | **0.683** | **+3.4%p** over baseline |
| **GT letter L20 (early emergence)** | delta_direct 0.531 vs baseline 0.504 | +2.7%p over baseline |

→ **Explicit direction-related supervision이 letter mapping을 추가로 향상시킨다.** Projector는 방향을 유지한 채 약간 "선명하게" 조정되고, LLM은 direction→letter pathway를 더 명확히 학습.

## 7.4 그림으로 이해하는 차이

```
Vision tokens          Projector         LLM (Qwen-7B)              Output
───────────────        ──────────        ───────────────────        ──────

Vanilla:
 ┌─pos info (R²=0.81)  ┌─ [frozen]      ┌─ L1-L4: direct info
 │ dir info (0.87)     │  base weight   │     extract ✓
 │ obj info (0.80)     │                │  L4-L28: info decays ↓
 └────────────────     └─ [frozen]      │  L28: dir 0.58 ⚠
                                        └─ Letter L28: 0.30 ✗ (chance 0.25)

Baseline (LoRA only):
 (same as vanilla)     (≈ vanilla)     ┌─ L1-L4: same as vanilla
                                        │  L8-L28: dir preserved
                                        │       0.82-0.87 ✓✓
                                        └─ Letter L20→L28: 0.50→0.65 ✓

Ours (delta_direct/channel_gate):
 (same inputs)         ┌─ proj tuned    ┌─ L1-L4: stronger dir
                       │  5% weight     │       0.77-0.79
                       │  row_cos>0.98  │  L8-L28: dir preserved
                       │                │       0.86-0.89 ✓✓
                       └─ F_top ↑       └─ Letter L20→L28: 0.53→0.68 ✓✓
```

---

# 8. Method design implications

## 8.1 Representation-level 원리

1. **Vision token에 정보는 이미 있다** — direction, position, semantic이 distributed하지만 linearly decodable
2. **Position/Direction과 Semantic은 분리된 subspace** — 한쪽 조작이 다른쪽을 손상시키지 않음
3. **Complexity는 signal 희석**: 배경 복잡도 주범 → real-world deploy 시 이 점 고려

## 8.2 LLM propagation 원리

4. **Vanilla LLM은 vision info를 layer 1–4에서 최대 수용, 이후 감쇠**
5. **LoRA는 "vision-to-language" attention pattern을 주로 조정**해 **깊은 layer direction 보존력**과 **letter mapping**을 부여
6. **Direction-specific supervision (delta_direct)**은 "direction information → letter output" 매핑을 더 직접적으로 강화

## 8.3 실제 method 설계 가이드라인

- **Projector 전체 weight를 크게 바꿀 필요 없음** (5% 수준으로 충분)
- **Row cosine을 보존하는 조정**이 semantic 유지하면서 direction만 강화 가능
- **얕은 layer에서 vision info를 더 aggressive하게 aggregate**하도록 design하면 효과적 (L1–L4)
- **Output side (LM head 근처)에서 explicit supervision**이 generation 성능에 직접 기여 (4-way에서 letter mapping은 L20부터 emerge하고 L28에서 peak; vanilla는 L28에서도 chance 근방)
- Real-world 조건에서 ours의 효과가 제한적이면, **training data를 복잡 조건 (obj_place 유사)으로 확장**하는 것이 유효할 것

---

# 9. 부록: Implementation Details (per experiment)

## Shared infrastructure

- **GPU**: 6× A100 80GB, fp16 inference
- **sklearn**: v1.2.2 (standard scaler, CV)
- **scipy.stats**: f_oneway for ANOVA
- **torch**: 2.9.0+cu128
- **Paths**:
  - Data: `/local_datasets/vlm_direction/vlm_direction_testbed/R2R_video_1500/`
  - MCQ JSON: `/data/jongseo/project/vlm/LLaVA-NeXT/dataset/R2R_4way_1500_json/`
  - SigLip: `google/siglip-so400m-patch14-384`
  - Base LLM: `lmms-lab/LLaVA-Video-7B-Qwen2`
  - Output: `analysis/token_linear_probing_R2R_1500/`

## A (basic probes)

- Script: `scripts/probe_vanilla.py :: main_basic_probes`
- Input: `features/vanilla_obj_place.npz` (N=6000, T=8)
- Frame sampling: `np.linspace(0, 31, 8, dtype=int)` → 8 uniform frames from 32
- Scaler: `StandardScaler` fit per fold (train fit, test transform)
- Ridge: closed-form, α=1.0, bias column added but not regularized
- 5-fold CV (KFold for regression, StratifiedKFold for classification)
- Classification probe: `nn.Linear`, Adam lr=0.01, weight_decay=1e-4, 300 epochs full-batch

## B (subspace / null ablation)

- Script: `scripts/probe_vanilla.py :: main_subspace_ablation`
- PCA:
  - `sklearn.decomposition.PCA(n_components=1000)` on StandardScaled frame-level features
  - Subsample 15,000 frames (out of 48,000) for PCA fit, seed=42
  - `pca.explained_variance_ratio_` for variance reporting
- Null ablation:
  - Per-dim position correlation: `sqrt(corr_x² + corr_y²)` (StandardScaled X vs (pos_x, pos_y))
  - `top_pos[k] = np.argsort(-corr_pos)[:k]`
  - `remaining = sorted(set(range(D)) − set(top_pos[k]))`
  - Object probe at mid frame (t=4), direction probe via delta (t=7 − t=0)
  - k ∈ {0, 10, 50, 100, 200, 500}

## C (temporal probes)

- Script: `scripts/probe_vanilla.py :: main_basic_probes` (direction 부분)
- Single frame: `feat[:, 4, :]` (index 4 out of 8 sampled)
- Delta: `feat[:, -1, :] - feat[:, 0, :]`
- T-mean: `feat.mean(axis=1)`
- Stack: `feat.reshape(N, 8*D)` — D = 1152 for pre, 3584 for post
- All use 4-way LogisticRegression via `cv_cls` (300-epoch Adam)

## D (dim overlap)

- Script: `scripts/probe_vanilla.py :: main_dim_overlap`
- Top-k based on:
  - Position: `sqrt(corr_x² + corr_y²)` on frame-level 15,000 subsample
  - Direction: ANOVA F-stat over 4 groups on delta (N=6000)
  - Object: ANOVA F-stat over 26 groups on mid frame (N=6000)
- k = 50
- Output: per-dim stats saved in `results/dim_stats_obj_place_{pre,post}_proj.npz`

## E (complexity gradient)

- Script: `scripts/probe_vanilla.py :: main_complexity_gradient`
- Balanced subsample: 375 per direction × 4 = 1500 per condition (common N for fair comparison)
- Probes: same as Section A/C
- Output: `results/complexity_gradient.json`

## F (trained model comparison)

- Script: `scripts/extract_multiproj.py` + `scripts/probe_trained.py`
- Extraction (`extract_multiproj.py`):
  - 6-GPU parallel (1000 videos per worker)
  - Each worker loads SigLip + 4 projectors (vanilla + 3 trained)
  - Per video: SigLip forward → each projector per-patch forward → mean-pool each
  - Output fp16: `features/multiproj_obj_place.npz` (~1.3 GB)
- Probing (`probe_trained.py`):
  - Same linear probe setup as Section A
  - Per-dim F-stat on delta
  - Top-50 dim analysis:
    - `vanilla_top50 = argsort(-vanilla_f_dir)[:50]`
    - `overlap_w_vanilla = |trained_top50 ∩ vanilla_top50|`
    - `strengthened = sum(trained_f_dir[vanilla_top50] > vanilla_f_dir[vanilla_top50])`

## G (projector weight)

- Script: `scripts/probe_trained.py :: compute_projector_weight_diff`
- Vanilla weights loaded from `model.safetensors.index.json` → filter `mm_projector.*`
- Trained weights from `{model_dir}/non_lora_trainables.bin` (same key filter)
- Cast to fp32 for stable norm/cosine
- `rel_diff = ||W_t - W_v|| / ||W_v||` (Frobenius)
- `row_cos = F.cosine_similarity(W_v, W_t, dim=1)` (per output dim)

## H (LLM hidden state)

- Scripts: `scripts/extract_llm_hidden.py` + `scripts/probe_llm_hidden.py`
- Extraction:
  - 4-GPU parallel (1 model per GPU)
  - Manual LoRA merge (bypass PEFT bnb issue):
    ```python
    for each layer in adapter_model.safetensors:
        A = lora_A.weight   # (r, in)
        B = lora_B.weight   # (out, r)
        W_base += (α/r) * (B @ A)
    ```
  - Forward pass:
    ```python
    (_, _, attn, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(...)
    outputs = model.forward(inputs_embeds=inputs_embeds, attention_mask=attn,
                            output_hidden_states=True, return_dict=True)
    # hs[0] = embedding, hs[k>0] = after block k-1
    last_token_hidden = hs[k][0, -1, :]
    ```
  - Layers probed: [0, 1, 4, 8, 12, 16, 20, 24, 28]
  - Output fp16: `features/llm_hidden_obj_place_{model}.npz`
- Subsample: 500 per direction × 4 = 2000 balanced (seed=42, `np.random.RandomState(42)`)
- Prompt tokenization:
  - **Critical**: `tokenizer_image_token` from `llava.mm_utils` (not plain tokenizer) — this replaces `<image>` with `IMAGE_TOKEN_INDEX` needed by `prepare_inputs_labels_for_multimodal`
  - Conv template: `qwen_1_5`
- Probes (`probe_llm_hidden.py`):
  - Direction: 4-way (from metadata)
  - GT Letter: `candidates.index(dir_text_map[direction])` — 4-way index (UDLR shuffled)
  - Object (for obj_* conditions): 26-way `obj_class`

## 전체 runtime

| 단계 | 시간 | 비고 |
|---|---|---|
| Vanilla extraction (4 conditions) | ~10 min | 6-GPU parallel, shape_* ~2.5 min each |
| Multi-projector extraction (obj_place) | ~5 min | 6-GPU parallel |
| LLM hidden extraction (2000 vids × 4 models) | ~10 min | 4-GPU parallel |
| Vanilla probes | ~3 min | GPU 4 |
| Trained probes | ~2 min | GPU 4 |
| LLM probes | ~3 min | GPU 0 |
| **Total** | **~35 min** | |

---

# 10. 파일 구조

```
analysis/token_linear_probing_R2R_1500/
├── PLAN.md                            # Experiment plan (pre-execution)
├── FINAL_REPORT.md                    # ← THIS DOCUMENT
│
├── scripts/
│   ├── extract_vanilla.py             # A, B, C, D, E — vanilla pre/post extract (4 conds × 6000)
│   ├── extract_multiproj.py           # F, G — multi-proj per-patch extract (obj_place × 6000)
│   ├── extract_llm_hidden.py          # H — LLM hidden state extract (4 models × 2000)
│   ├── probe_vanilla.py               # A, B, C, D, E — all vanilla probes
│   ├── probe_trained.py               # F, G — trained model comparison
│   └── probe_llm_hidden.py            # H — LLM layer-wise probing
│
├── features/  (~3 GB total)
│   ├── vanilla_shape_color.npz        # (6000, 8, 1152) + (6000, 8, 3584)
│   ├── vanilla_shape_place.npz
│   ├── vanilla_obj_color.npz
│   ├── vanilla_obj_place.npz          # Primary
│   ├── multiproj_obj_place.npz        # 4 different post-proj versions (1.3 GB)
│   └── llm_hidden_obj_place_{vanilla,baseline,delta_direct,channel_gate}.npz
│
├── results/
│   ├── basic_probes_obj_place.json    # Section A
│   ├── subspace_obj_place.json        # Section B
│   ├── dim_overlap_obj_place.json     # Section D
│   ├── dim_stats_obj_place_{pre,post}_proj.npz    # Per-dim F-stats
│   ├── complexity_gradient.json       # Section E
│   ├── trained_comparison_obj_place.json          # Section F, G
│   ├── fstats_obj_place_{vanilla,baseline,delta_direct,channel_gate}.npz  # Section F
│   └── llm_probe_obj_place.json       # Section H
│
└── logs/
    ├── extract_vanilla.log
    ├── extract_multiproj.log
    ├── extract_llm.log
    ├── probe_vanilla.log
    ├── probe_trained.log
    └── probe_llm.log
```
