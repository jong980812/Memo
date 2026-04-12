# Vision Token 정보 분석: LLaVA-Video의 Position·Direction·Semantic 인코딩 해부

> **통합 보고서** — LLaVA-Video-7B-Qwen2의 vision tokens에 어떤 정보가 어떻게 인코딩되어 있는지, 그리고 어떻게 LLM에 전달되는지를 체계적으로 분석한 프로젝트의 최종 보고서.

---

# 목차

1. [연구 동기와 핵심 질문](#1-연구-동기와-핵심-질문)
2. [배경 지식](#2-배경-지식)
3. [모델과 데이터](#3-모델과-데이터)
4. [용어 정리 (Glossary)](#4-용어-정리-glossary)
5. [방법론: Linear Probing](#5-방법론-linear-probing)
6. **실험 결과**
   - [Part A. Mean-pooling 후에도 위치 정보가 살아있는가?](#part-a-mean-pooling-후에도-위치-정보가-살아있는가)
   - [Part B. 위치 정보는 어디에 있나? — Subspace 분석](#part-b-위치-정보는-어디에-있나--subspace-분석)
   - [Part C. 움직임 정보는 어떻게 표현되는가?](#part-c-움직임-정보는-어떻게-표현되는가)
   - [Part D. Position, Direction, Semantic의 관계 — Dim Overlap 분석](#part-d-position-direction-semantic의-관계--dim-overlap-분석)
   - [Part E. Projector는 무엇을 바꾸나?](#part-e-projector는-무엇을-바꾸나)
   - [Part F. 시각적 복잡도가 direction encoding에 미치는 영향](#part-f-시각적-복잡도가-direction-encoding에-미치는-영향)
   - [Part G. 실제 영상에서도 작동하나? — Real-world 검증](#part-g-실제-영상에서도-작동하나--real-world-검증)
   - [Part H. LLM은 vision 정보를 가져가는가?](#part-h-llm은-vision-정보를-가져가는가)
   - [Part I. Fine-tuned 모델은 무엇이 달라지나? (Vanilla vs baseline vs delta_direct vs channel_gate)](#part-i-fine-tuned-모델은-무엇이-달라지나-vanilla-vs-baseline-vs-delta_direct-vs-channel_gate)
7. [종합 발견 사항 (Summary of Findings)](#7-종합-발견-사항-summary-of-findings)
8. [Method Design에의 시사점](#8-method-design에의-시사점)
9. [부록: Implementation Details](#9-부록-implementation-details)
10. [파일 구조](#10-파일-구조)

---

# 1. 연구 동기와 핵심 질문

## 문제 설정
Video LLM(LLaVA-Video)은 영상의 "움직임 방향"이나 "물체 위치" 같은 질문에서 성능이 낮다. 우리가 제안한 method들 (delta_direct, channel_gate)은 이 문제를 개선하는데, **왜 개선되는지**를 representation level에서 이해하고 싶다.

그러기 위해서는 먼저 vanilla LLaVA-Video의 vision representation이 **어떤 정보를, 어떤 dimension에, 어떻게** 담고 있는지부터 파악해야 한다.

## 핵심 연구 질문

1. **Q1 (Position)**: Vision encoder(SigLip)가 만드는 729개 패치 토큰을 N축으로 평균낸 D-dim 벡터에 위치 정보가 살아있는가?
2. **Q2 (Subspace)**: 그 위치 정보가 특정 소수 dim에 집중되어 있나, 아니면 전체 D에 분산되어 있나?
3. **Q3 (Motion)**: 한 프레임만으로는 방향을 모르는 게 당연한데, 여러 프레임을 쌓으면 움직임이 decode 가능한가?
4. **Q4 (Separation)**: 위치·방향·semantic 정보가 서로 같은 dim에 섞여 있나, 분리되어 있나?
5. **Q5 (Projector)**: mm_projector를 통과하면 이 정보 구조가 어떻게 바뀌나?
6. **Q6 (Complexity)**: 배경이 복잡해지거나 물체가 실제 object가 되면 direction encoding이 약해지나?
7. **Q7 (Real-world)**: 합성 영상이 아닌 실제 영상(SSv2, KTH)에서도 같은 현상이 관찰되나?
8. **Q8 (LLM)**: LLM을 통과하면서 vision 정보가 text token(특히 last token)에 얼마나 전달되나? 어느 layer가 가장 정보를 많이 담나?
9. **Q9 (Fine-tuning)**: 학습된 모델(baseline, delta_direct, channel_gate)은 vanilla와 비교해 무엇이 바뀌었나? 정보가 새로 생긴 건가, 기존 dim이 강화된 건가?

---

# 2. 배경 지식

## 2.1 Vision Transformer with Absolute Position Embedding

SigLip ViT는 이미지를 384×384로 받고, 14×14 패치로 나눠서 **27×27 = 729개 패치**로 만든다. 각 패치는 독립적으로 embedding된 뒤 **absolute position embedding**이 더해진다:

```
token_i = patch_embed(patch_i) + position_embed(i)
```

여기에 27개의 self-attention layer가 쌓인다.

### Equivariance 깨짐
만약 position embedding이 없었다면 (CNN처럼), 물체가 왼쪽에 있든 오른쪽에 있든 같은 패치가 같은 값을 갖고 위치만 달라짐 (translation equivariance). 이 경우 패치 평균을 내면 같은 위치 분포의 영상끼리는 구별이 안 된다.

**하지만 absolute position embedding이 더해지면**:
- 왼쪽에 있는 물체: 왼쪽 position_embed + object_embed
- 오른쪽에 있는 물체: 오른쪽 position_embed + object_embed
- 그리고 self-attention이 이를 **모든 패치로 전파**함

→ 결과적으로 "물체가 어디에 있느냐"에 따라 **모든 패치의 값이 달라지고**, 이게 mean-pool 후에도 D-dim 벡터에 남아있을 가능성이 있다.

이게 Q1의 핵심 가설이다.

## 2.2 LLaVA-Video Pipeline

```
Frames (8, 3, 384, 384)
   ↓ SigLip ViT (frozen, 27 layers with abs pos_emb)
Vision tokens (8, 729, 1152)        [pre-projection]
   ↓ mm_projector (mlp2x_gelu)
Projected tokens (8, 729, 3584)      [post-projection]
   ↓ merge with text tokens
Input embeddings → Qwen2-7B LLM (28 decoder layers)
   ↓ 
Hidden states at each layer × last token (3584-d)
   ↓ LM head
Next token logits
```

- **mm_projector 구조**: `Linear(1152→3584) → GELU → Linear(3584→3584)`
- Patch token 단위로 적용됨 (patch별로 독립적)
- Trained model에서는 이 projector + LoRA-applied LLM이 학습됨 (SigLip은 frozen)

---

# 3. 모델과 데이터

## 3.1 사용한 모델
- **Vanilla**: `lmms-lab/LLaVA-Video-7B-Qwen2` (base model, 학습 없음)
- **Trained**: `4combo_new/work_dirs/` 아래
  - `baseline` (LoRA on LLM + mm_projector trained)
  - `delta_direct` (LoRA + direction loss 추가)
  - `channel_gate` (LoRA + channel attention 모듈 추가)
- 셋 다 `shape_simple_new` 데이터셋으로 학습

## 3.2 사용한 데이터셋

### Synthetic Testbed
모든 합성 영상은 **384×384** 해상도, 단색 배경 혹은 실제 장소 배경, 물체가 4방향(up/down/left/right) 중 하나로 직선 운동.

| 이름 | 위치 | 크기 | 프레임 | 배경 | 물체 | 용도 |
|---|---|---|---|---|---|---|
| `R2R_*_video` (testbed) | `/local_datasets/vlm_direction/vlm_direction_testbed/R2R_4way_video/` | 각 200 videos × 4 conditions | 32 frames | 단색/실제장소 | 도형/COCO obj | Complexity gradient 분석 |
| `R2R_real_color` (메인) | `/local_datasets/vlm_direction/R2R_real_color/` | **8000 videos** | 8 frames | 단색 | 20 COCO objects | 주력 분석 데이터 (20×4×100) |

### Testbed 4 conditions (각 200 videos)
- `shape_color`: 도형 × 단색 배경 (가장 단순)
- `shape_place`: 도형 × 실제 장소 배경
- `obj_color`: 실제 COCO 물체 × 단색 배경
- `obj_place`: 실제 물체 × 실제 장소 배경 (가장 복잡)

### R2R_real_color (메인 데이터셋)
- **8000 videos** = 20 objects × 4 directions × 100 instances
- Objects: car, chair, dining table, person, cup, bowl, bottle, book, truck, motorcycle, umbrella, broccoli, banana, bench, sheep, potted plant, handbag, cake, cow, boat
- 각 물체마다 4방향 × 100개 (방향별 2000개)
- 8 frames, 384×384, 단색 배경, random start/end positions, linear motion
- Metadata: `id`, `direction`, `object`, `start_pos`, `end_pos`, `positions` (frame별 ground truth)

### Real-world Datasets
| 이름 | 위치 | 크기 | Task |
|---|---|---|---|
| SSv2 VP | `/local_datasets/vlm_direction/ssv2_VP/ssv2_VP_default/` | left/right 각 250 = 500 | Something moving left/right |
| KTH VP | `/local_datasets/vlm_direction/KTH_VP_lr/KTH_VP_default/` | left/right 각 ~450 = 899 | Person walking/jogging left/right |

---

# 4. 용어 정리 (Glossary)

## Token / Feature 용어

| 용어 | 의미 |
|---|---|
| **Patch token** | SigLip이 이미지를 14×14로 나눈 각 조각에 해당하는 벡터. 총 729개 (=27×27). |
| **Pre-projection token** | SigLip 마지막 hidden state. Shape: (B, 729, **1152**) |
| **Post-projection token** | mm_projector를 통과한 후. Shape: (B, 729, **3584**) |
| **Mean-pooled token** | 729개 패치를 평균낸 벡터. `pre_mean`: (T, 1152), `post_mean`: (T, 3584) |
| **LLM hidden** | LLM을 통과한 후 각 layer의 last token 위치의 hidden state. 3584-d |
| **Frame-level sample** | 한 프레임의 mean-pooled 벡터 하나 = 1 sample |
| **Video-level sample** | 한 영상 전체에서 derived된 벡터 하나 (e.g., delta, mid frame, T-mean) = 1 sample |

## Probe Target 용어

| 용어 | 의미 |
|---|---|
| **Position (x, y)** | 해당 프레임에서 물체의 픽셀 좌표. Metadata `positions`에서 읽음. |
| **Direction** | 영상 전체의 이동 방향 (4-way: up/down/left/right, 또는 2-way: left/right) |
| **Object class** | 20 COCO 물체 중 하나 (R2R_real_color), 또는 26 클래스 (testbed) |
| **Quadrant** | 화면을 중심(192,192) 기준 4사분면으로 나눠 물체가 어느 사분면에 있는지 |

## 파생 벡터 용어

| 용어 | 정의 | 용도 |
|---|---|---|
| **Single frame** | `feat[:, t_mid, :]` (mid frame 하나) | Object class, Direction sanity check |
| **Delta** | `feat[:, -1, :] - feat[:, 0, :]` (마지막 - 첫 프레임) | Direction에 민감 |
| **T-mean** | `feat.mean(axis=1)` (시간축 평균) | 방향 정보 제거, object 강조 |
| **Temporal stack** | `feat.reshape(N, T*D)` (모든 프레임 펼침) | Full temporal info |

## 지표 용어

| 용어 | 의미 |
|---|---|
| **R² (R-squared)** | Regression이 실제 값 분산의 몇 %를 설명하는지 (1.0 = 완벽, 0 = chance). |
| **F-statistic (F-stat)** | One-way ANOVA 통계량. 클래스 간 분산 / 클래스 내 분산. 해당 dim이 클래스를 얼마나 구별하는지. |
| **Top-50 dim** | 특정 task(direction, object 등)에 대해 F-stat이 가장 높은 50개 dim index. |
| **Overlap k/50** | 두 집합(예: pos top-50 vs dir top-50)이 공유하는 dim 수. |
| **Cross-probe** | A의 top-k dim만 써서 B를 예측하는 실험 (가령 position dim으로 object 예측). |

---

# 5. 방법론: Linear Probing

## 기본 아이디어
"특정 representation이 어떤 정보를 담고 있는지"를 측정하는 방법은 여러 가지가 있는데, 여기서는 **Linear Probing**을 사용한다. 이유:
- **단순성**: linear model(Ridge regression 또는 Logistic regression)만 사용
- **Interpretability**: 선형으로 풀린다 = 정보가 깔끔하게 decodable하게 인코딩되어 있다
- **공정성**: 모델 간 비교 시 probe model capacity가 고정

## 구체적 방법

### Regression (R²)
```python
# Ridge closed-form: w = (X'X + αI)^{-1} X'y
# 5-fold CV, α=1.0
r2 = 1 - (sum((y - pred)²) / sum((y - y.mean())²))
```

### Classification (Accuracy)
```python
# nn.Linear + Adam (lr=0.01, weight_decay=1e-4)
# 300 epochs, full-batch, 5-fold StratifiedKFold
acc = (model(X_test).argmax(1) == y_test).mean()
```

### F-statistic per dim
```python
# 각 dim d에 대해 one-way ANOVA
# 4 direction groups [X[dir==0, d], X[dir==1, d], ..., X[dir==3, d]]
# 클래스 간 분산 / 클래스 내 분산
f_stat[d] = f_oneway(groups).statistic
```

Higher F-stat → 그 dim이 class를 더 잘 구별함.

---

# Part A. Mean-pooling 후에도 위치 정보가 살아있는가?

## A.1 실험 설정
- **데이터**: R2R_real_color (8000 videos × 8 frames = 64000 frame samples)
- **Input**: pre-projection mean-pooled token (1152-d) 또는 post-projection (3584-d)
- **Target**: 그 프레임에서의 물체 (x, y) 픽셀 좌표
- **Method**: Ridge regression (α=1.0), 5-fold CV
- **Chance baseline**: R² ≈ 0

## A.2 결과

### R2R_real_color (8000 videos, 64000 frame samples)

| Metric | Pre-proj (1152-d) | Post-proj (3584-d) | Chance |
|--------|-------------------|--------------------|--------|
| **Position R² (x)** | **0.727** | **0.806** | 0 |
| **Position R² (y)** | **0.895** | **0.933** | 0 |
| Quadrant acc (N=6400 frames) | 0.961 | - | 0.25 |

### Testbed (가장 단순 조건, 200 videos)

| 조건 | Pre-proj R²(x) | R²(y) |
|---|---|---|
| shape_color (도형 × 단색) | **0.986** | **0.996** |
| obj_color | 0.889 | 0.964 |
| obj_place | 0.925 | 0.942 |

## A.3 해석

**Q1 ("mean-pool 후 위치 정보 남나?")에 대한 답: 압도적으로 YES.**

- R2R_real_color에서도 R²(x) = 0.73, R²(y) = 0.90 — 단순 Ridge로 위치 복원 가능
- Testbed의 가장 깔끔한 조건(shape_color)에서는 R² = 0.99 (거의 완벽)
- **이는 ViT의 absolute position embedding + self-attention propagation이 위치 정보를 모든 패치 값에 남기기 때문**

**보충: Projector 통과 후 오히려 강화됨** (0.727 → 0.806 for x). mm_projector가 위치 encoding을 정제한다.

### 왜 x가 y보다 낮은가?
R²(y) > R²(x)인 현상이 일관되게 관찰됨. 가설:
- 물체 크기/가로폭 분산이 y축 encoding에는 영향을 덜 줌
- 아니면 SigLip의 position embedding이 y축 방향에 더 discriminative하게 학습됨

---

# Part B. 위치 정보는 어디에 있나? — Subspace 분석

## B.1 동기
"위치 정보가 1152-d 벡터 어딘가에 있다"는 건 알겠는데, 구체적으로 **어떻게** 있을까?
- 가설 1: 특정 소수 dim에 집중 (low-rank subspace)
- 가설 2: 전체 D에 분산

이걸 구별하면 method design에 직접 영향을 준다.

## B.2 실험 B1: PCA Dimensionality
- **Features**: pre-projection (1152-d)
- **Method**: 10000 frame subsample → PCA → 상위 k개 PC만 써서 Ridge R²
- k ∈ {1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000}

| PCA dims (k) | Variance explained | R²(x) | R²(y) |
|---|---|---|---|
| 5 | 64.7% | 0.004 | 0.015 |
| 10 | 69.4% | 0.005 | 0.059 |
| 50 | **81.7%** | **0.060** | **0.493** |
| 100 | 86.7% | 0.123 | 0.691 |
| 200 | 91.6% | 0.285 | 0.782 |
| 500 | 97.1% | 0.522 | 0.843 |
| 1000 | 99.7% | 0.633 | 0.870 |
| 1152 (full) | 100% | **0.727** | 0.895 |

**핵심 관찰**: PCA 50개는 variance의 82%를 설명하지만, position R²(x)는 겨우 0.06. 대부분의 분산을 차지하는 PC들이 **위치 정보를 거의 안 담고 있다**. 위치를 복원하려면 뒤쪽 PC들까지 합쳐야 함 → **위치 정보는 전체 D에 분산**.

## B.3 실험 B2: Position-nulled Ablation
"Top-k position-correlated dim을 제거하면 position/direction/object가 얼마나 망가지나?"

- **Top position dim 선정 기준**: per-dim `sqrt(corr_x² + corr_y²)` 상위 k개
- **Remaining dims**: 나머지 (1152 - k) dim만 써서 다시 probe

| 제거한 pos dim 수 | Object class (20) | Direction (delta) | Position R²(x) |
|---|---|---|---|
| 0 (full) | 0.927 | 0.912 | 0.73 |
| 10 | 0.925 | 0.911 | 0.68 |
| 50 | 0.927 | 0.900 | 0.60 |
| 100 | 0.926 | 0.897 | 0.59 |
| **200** | **0.924** | **0.886** | **0.56** |

**관찰**:
- Top-200 position dim을 통째로 빼도 position R² = 0.56 (나머지 952 dim에 위치 정보 남아있음)
- **Object classification은 전혀 영향 없음** (0.927 → 0.924)
- Direction은 미미한 하락 (0.912 → 0.886)

## B.4 B 종합 결론

1. **Q2 ("특정 subspace인가")에 대한 답: NO. 위치 정보는 전체 1152-d에 분산 인코딩되어 있다.**
2. **Top-k position dim과 Object dim은 완전히 다름** (object는 top-200 제거에도 무영향) → 다음 Part D에서 자세히
3. 이는 **"위치 정보를 담당하는 몇 개 뉴런이 있다"는 단순한 그림이 아님**을 의미. 모든 dim이 약하지만 일관되게 위치 정보를 담고 있다.

---

# Part C. 움직임 정보는 어떻게 표현되는가?

## C.1 동기
Position 정보는 frame 단위 정적 정보. **움직임(direction)**은 시간축이 필요한 정보. 어떻게 decode할 수 있나?

## C.2 파생 벡터 실험

4가지 파생 방법을 비교:

| 파생 방법 | 수식 | Sample 단위 | 시간 정보 |
|---|---|---|---|
| Single frame | `feat[:, t_mid, :]` | video | 없음 (한 프레임만) |
| T-mean | `feat.mean(axis=1)` | video | 시간 순서 제거, 평균 위치만 |
| Delta (last-first) | `feat[:, -1] - feat[:, 0]` | video | 위치 변화량 |
| 8-frame stack | `feat.reshape(N, 8*D)` | video | 전체 시간 정보 |

## C.3 결과 (R2R_real_color, 8000 videos, pre-proj)

| 방법 | Direction acc | Chance |
|---|---|---|
| Single frame (mid) | 0.408 | 0.25 |
| T-mean | 0.358 | 0.25 |
| **Delta (last - first)** | **0.912** | 0.25 |
| 8-frame stack | 0.820 | 0.25 |

## C.4 해석

### Single frame 40.8%: Sanity check + 데이터 편향
- 단일 프레임만으로는 이론적으로 방향을 알 수 없어야 함 (어디로 가는지 미래 정보 필요)
- 그런데 chance(25%) 위의 40.8%는 **위치 편향 때문**: 8000 samples에서 linear model이 "mid frame에서 x가 크면 right 방향일 확률 높음" 같은 패턴을 학습함 (R2R은 random start/end이지만 샘플 수 많으면 약한 bias 잡힘)

### T-mean 35.8%: 시간 순서 제거 확인
- 8 프레임의 평균: 궤적의 "중심점"만 남김
- 방향 정보가 **대부분 사라짐** (chance 근처) — 시간 순서가 방향의 핵심이라는 증거

### **Delta 91.2%: 핵심 결과**
- 두 프레임의 mean-pooled 벡터 차이만으로 방향을 거의 완벽히 예측
- 왜 가능한가? Position이 각 dim에 linear하게 인코딩되어 있으므로:
  - `pre_mean_t = W · pos_t + 기타 정보`
  - `delta = pre_mean_T - pre_mean_0 = W · (pos_T - pos_0) + (기타 noise)`
  - → delta에 위치 변화량(= 속도 벡터)이 그대로 남음
- 이는 **"mean-pooled vector 2개만으로 움직임 모델링 가능"**하다는 강력한 증거

### Temporal stack 82%: 더 많은 정보 but 약한 probe
- 8개 프레임 전체를 사용 (sample당 9216-d feature)
- 8000 samples로 충분히 학습되지만 delta보다 낮음
- 이유: feature dim이 크면 linear probe의 bias/variance 트레이드오프 불리
- 정보량 자체는 delta보다 많지만, linear decoder가 따라가지 못함

## C.5 결론
**Q3 ("시간축으로 쌓으면 방향 decode 가능?")에 대한 답: YES, 특히 delta로 매우 강하게 (91%).**

(T, D) 벡터 시퀀스는 방향 정보를 풍부하게 담고 있다. 그리고 이 정보의 source는 Part A에서 확인한 "각 frame의 위치 정보가 D-dim 벡터에 linear하게 인코딩되어 있다"는 사실이다.

---

# Part D. Position, Direction, Semantic의 관계 — Dim Overlap 분석

## D.1 동기
같은 D-dim 벡터 안에:
- 위치 정보
- 방향 정보
- 물체 종류 정보

가 모두 들어있는데, **같은 dim을 공유하나, 아니면 분리된 dim에 들어있나**?

## D.2 Top-50 dim 선정 방법

| Task | Top-50 dim 선정 기준 |
|---|---|
| Position | Frame-level feature에서 per-dim `sqrt(corr_x² + corr_y²)` 상위 50 |
| Direction | Delta vector에서 4-group one-way ANOVA F-stat 상위 50 |
| Object | Mid frame feature에서 20-group ANOVA F-stat 상위 50 |

## D.3 Overlap 결과 (R2R_real_color, 8000 videos)

| Overlap | Pre-proj (1152-d) | Post-proj (3584-d) | Random expectation |
|---------|-------------------|--------------------|--------------------|
| **Position ∩ Direction** | **37/50** | **30/50** | 2.2 / 0.7 |
| **Position ∩ Object** | **3/50** | **0/50** | 2.2 / 0.7 |
| **Direction ∩ Object** | **2/50** | **1/50** | 2.2 / 0.7 |

## D.4 Cross-probe 결과
"A dim 집합(top-50)만 써서 B를 예측"

| Probe | Pre-proj | Post-proj | Chance |
|-------|----------|-----------|--------|
| Object from **position dims** | 0.704 | 0.749 | 0.05 |
| Object from **object dims** | 0.767 | 0.759 | 0.05 |
| Direction from **direction dims** | 0.757 | 0.643 | 0.25 |
| Direction from **position dims** | 0.746 | 0.631 | 0.25 |

## D.5 해석

### 1) Position ≈ Direction (같은 dim 사용)
- 37/50 (pre), 30/50 (post) overlap — random 기대값 2.2/0.7 대비 매우 높음
- **왜?** Direction = Position의 시간 변화. 위치 정보를 담는 dim의 delta가 direction을 담는다. 따라서 동일 dim.

### 2) Position/Direction vs Object는 분리
- Overlap 3/50, 0/50 — 거의 random 수준 또는 완전히 0
- **가장 강한 signal을 담는 top-50 dim이 서로 다름**
- 즉 "위치/방향 담당 dim"과 "semantic 담당 dim"이 분리되어 있다

### 3) Cross-probe는 왜 높게 나오나? (70% object from position dims)
- Top-50 primary dim이 겹치지 않아도, **약한 signal은 곳곳에 퍼져 있음**
- 50개 dim이면 8000 samples에 충분히 많아서 logistic regression이 약한 신호까지 다 모아서 예측
- 결론: **"Primary encoding은 분리, 약한 secondary signal은 공존"**

### 4) Post-projection에서 더 깔끔하게 분리
- Pos-Obj: 3/50 → 0/50
- mm_projector가 position/semantic 정보를 **더 분리해서 LLM에 전달**

---

# Part E. Projector는 무엇을 바꾸나?

## E.1 Pre-proj (1152-d) vs Post-proj (3584-d) 성능 비교

### R2R_real_color (8000 videos)

| Metric | Pre-proj | Post-proj | 변화 |
|---|---|---|---|
| Position R²(x) | 0.727 | 0.806 | **+0.08 ↑** |
| Position R²(y) | 0.895 | 0.933 | **+0.04 ↑** |
| Delta direction | 0.912 | 0.882 | -0.03 |
| Object class | 0.927 | 0.883 | -0.04 |
| Object T-mean | 0.946 | 0.909 | -0.04 |

### Overlap (Position/Direction vs Object)
| | Pre | Post |
|---|---|---|
| Pos ∩ Dir | 37/50 | 30/50 |
| Pos ∩ Obj | 3/50 | **0/50** |
| Dir ∩ Obj | 2/50 | 1/50 |

## E.2 해석

Projector는:
1. **Position encoding 정제**: R²(x) 0.73 → 0.81 (8%p 향상)
2. **Position-Object 분리 강화**: overlap 3/50 → 0/50 (완전 분리)
3. **Direction/Object는 소폭 감소**: 0.03-0.04 하락 (dim 늘어났는데 probe가 못 따라가는 효과 + 약간의 압축 loss)

**LLM 입장에서 중요한 건 "정보가 잘 분리된 형태로 오는 것"**. Projector는 semantic/spatial 정보를 명확히 구분해서 전달한다.

---

# Part F. 시각적 복잡도가 direction encoding에 미치는 영향

## F.1 동기
합성 영상(단색 배경 + 도형)에서는 direction encoding이 잘 되는 건 당연. **실제 물체와 실제 배경이 섞이면** direction dim이 약해질까?

## F.2 실험 설정
- **Testbed 4 conditions** (각 200 videos, 32 frames에서 8개 균등 샘플)
- **Features**: pre-projection
- **복잡도 gradient**: 단순 → 복잡

| 조건 | 배경 | 물체 | 복잡도 |
|---|---|---|---|
| shape_color | 단색 | 도형 | 최저 |
| obj_color | 단색 | 실제 COCO 물체 | 중 (물체 복잡) |
| shape_place | 실제 장소 | 도형 | 중 (배경 복잡) |
| obj_place | 실제 장소 | 실제 물체 | 최고 |

## F.3 결과

| 조건 | R²(x) | Delta dir | F_dir top50 평균 | dims F>5 |
|---|---|---|---|---|
| shape_color | **0.986** | **0.910** | 38.9 | **665** |
| obj_color | 0.889 | 0.885 | 18.4 | 337 |
| shape_place | 0.930 | **0.555** | 12.5 | **99** |
| obj_place | 0.925 | 0.700 | 13.1 | 116 |

## F.4 해석

### 배경 복잡도가 가장 큰 영향
- shape_color (단색 배경): Delta dir **0.91**
- shape_place (실제 배경): Delta dir **0.56** (↓ 35%p!)

**물체는 같은데 배경만 실제로 바꾸면 direction 성능이 절반 가까이 떨어진다.**

### 물체 복잡도는 상대적으로 영향 작음
- shape_color vs obj_color: Delta dir 0.91 → 0.89 (2%p만 하락)

### F-stat dim 수의 drastic한 감소
- shape_color는 F>5인 dim이 **665개** (절반 이상)
- shape_place는 **99개** (전체의 8.6%)
- → 실제 배경 추가 시 "direction을 강하게 구별하는 dim 수가 85% 감소"
- 배경이 random noise처럼 작용해 direction signal을 masking

## F.5 결론
**Q6 ("복잡도가 direction 약화시키는가")의 답: YES, 특히 배경 복잡도가 주범.**

이건 **method design에 중요한 힌트**: direction task 성능을 올리려면 **배경 noise를 suppress**하거나 **foreground object에 attention**을 집중시키는 mechanism이 유효할 것.

---

# Part G. 실제 영상에서도 작동하나? — Real-world 검증

## G.1 데이터
| Dataset | Task | Size | 특징 |
|---|---|---|---|
| **SSv2** (pushing) | left/right push | 500 | 손으로 물체 밀기. 진짜 motion 필요. 2-way, chance 50% |
| **KTH** (walking) | left/right walk | 899 | 사람이 걷거나 뜀. 자세로도 방향 암시됨. 2-way, chance 50% |

## G.2 결과

| Dataset | Stage | Delta dir | Single frame | T-mean | dims F>5 |
|---|---|---|---|---|---|
| **SSv2** (push) | pre | 0.662 | 0.452 | 0.440 | 49 |
|  | post | 0.666 | 0.448 | 0.466 | **441** |
| **KTH** (walk) | pre | 0.699 | **0.733** | 0.666 | 390 |
|  | post | 0.655 | 0.687 | 0.670 | **1400** |

## G.3 해석

### SSv2 (push motion)
- Single frame 0.45 (chance 근처) → push는 한 프레임만 봐서는 방향 모름 (정적인 물체+손)
- Delta 0.67 → motion으로부터 direction 추출 성공
- **진짜 motion-based direction encoding이 작동**

### KTH (walking)
- Single frame **0.73** → 사람 걷는 자세 자체가 방향을 암시 (얼굴 방향, 팔다리 자세)
- Delta와 T-mean도 비슷한 수준 → motion보다 자세가 더 informative
- **Direction이 motion이 아닌 static pose cue에 dominant**하게 실림

### Post-projection 효과
- SSv2: F>5 dim 수 49 → 441 (9배)
- KTH: 390 → 1400 (3.6배)
- Projector가 direction 신호를 **여러 dim으로 확산시킴** (합성 영상과 같은 패턴)

## G.4 결론
**Q7 ("실제 영상에서도 작동?")의 답: YES.**
- 동작 기반 (SSv2): delta로 잘 decode
- 자세 기반 (KTH): single frame으로도 잘 decode
- 합성 영상에서 관찰된 dim-level 현상이 실제 영상에도 그대로 나타남

---

# Part H. LLM은 vision 정보를 가져가는가?

## H.1 동기
지금까지 본 건 모두 "vision tokens에 정보가 있다". 하지만 **LLM을 통과하면서 이 정보가 text token(특히 마지막 token)에 얼마나 전달되나**? 이게 핵심이다. LLM이 generate할 때 참조하는 건 last token이니까.

## H.2 실험 설정
- **모델**: LlavaQwenForCausalLM (vanilla 7B, fp16)
- **입력**: 400 R2R_real_color videos (방향별 100개 balanced sampling)
- **Prompt**: `<image>\nDescribe the video.`
  - 중요: `tokenizer_image_token`을 써서 `<image>`를 IMAGE_TOKEN_INDEX로 변환해야 실제로 vision tokens가 병합됨
- **Forward pass**: `prepare_inputs_labels_for_multimodal` → `model.forward(inputs_embeds=..., output_hidden_states=True)`
- **추출**: 각 layer의 **last token hidden state** (3584-d)
- **Layers probed**: 0 (embedding), 1, 4, 8, 12, 16, 20, 24, 28 (total 28 decoder layers)
- **Probes**:
  - Direction: 4-way classification (linear + Adam)
  - Object: 20-way classification
  - Position R²(x): Ridge regression (target = mid frame 기준 x 좌표)

## H.3 결과: Last token이 layer별로 담는 정보

| Layer | Direction | Object | Position R²(x) |
|---|---|---|---|
| 0 (embedding) | 0.250 (chance) | 0.085 | -0.005 |
| **1** | **0.525** | **0.663** | 0.620 |
| **4** ★ | **0.573** | 0.645 | **0.846** |
| 8 | 0.480 | 0.603 | 0.839 |
| 12 | 0.443 | 0.485 | 0.666 |
| 16 | 0.483 | 0.498 | 0.609 |
| 20 | 0.463 | 0.483 | 0.598 |
| 24 | 0.463 | 0.478 | 0.586 |
| 28 (last) | 0.468 | 0.530 | 0.498 |

★ = peak

## H.4 해석

### Layer 0: Chance 수준 (당연)
- 모든 영상에 대해 prompt가 동일 → last token의 embedding도 동일
- Vision tokens은 merge되어 있지만, **last token 위치에서는 아직 attention으로 vision 정보가 흡수되지 않음**

### Layer 1: 즉시 흡수!
- Dir 0.25 → **0.525**, Obj 0.085 → **0.663**, Pos 0 → **0.620**
- **단 한 번의 self-attention block만 통과해도** last token이 vision token들의 정보를 끌어온다
- LLM이 vision 정보를 읽어가는 게 매우 빠르게 시작됨

### Layer 4: Peak
- Position R² **0.846** (가장 높음, Pre-proj vision token에서의 0.73보다도 높음)
- Direction **0.573**
- **이 layer가 vision 정보를 가장 raw하게 보존**

### Layer 4→28: 감쇠
- Position R² 0.85 → 0.50, Direction 0.57 → 0.47
- **LLM이 깊어질수록 정보가 task-specific representation으로 변환**되며 linear decode 가능한 raw 정보는 감소
- 하지만 **끝까지 chance 위 유지** — LLM generate 직전까지 vision 정보가 살아있음

## H.5 결론

**Q8 ("LLM이 vision 정보 가져가는가")의 답: YES, Layer 1부터 즉시. Layer 4가 peak.**

이건 **method design에 중요한 의미**를 가진다:
- **얕은 layer에서 vision supervision을 주는 게 효과적**일 수 있다 (Layer 1-4에서 정보가 풍부하니까)
- `llm_delta_direct`처럼 LLM 내부에서 supervision을 주는 기법이 이론적으로 뒷받침됨

---

# Part I. Fine-tuned 모델은 무엇이 달라지나? (Vanilla vs baseline vs delta_direct vs channel_gate)

## I.1 동기
이제까지는 vanilla 모델만 분석. 실제로 우리가 학습한 4개 모델은 vanilla보다 direction task 성능이 좋다. **무엇이 어떻게 달라져서 좋아진 것인가?**

## I.2 모델 개요
네 모델 모두:
- Vision encoder(SigLip)는 frozen (즉 pre-projection tokens은 동일!)
- mm_projector + LLM(LoRA)이 학습됨

| 모델 | 특징 |
|---|---|
| vanilla | 학습 없음 |
| baseline | 단순 LoRA 학습 |
| delta_direct | LoRA + explicit direction loss |
| channel_gate | LoRA + channel attention module |

## I.3 실험 설정
- **데이터**: R2R_real_color 8000 videos (동일)
- **Feature**: 4개 모델 각각의 **post-projection** 벡터 (3584-d, per-patch projection 후 mean-pool)
  - SigLip output은 공유, mm_projector만 모델별로 다름
- 모든 probe 방법은 이전 실험과 동일

## I.4 핵심 결과: Linear Probe Accuracy

| Model | Pos R²(x) | Pos R²(y) | Delta dir | Obj acc | F_dir top50 mean |
|---|---|---|---|---|---|
| vanilla | 0.691 | 0.895 | 0.884 | 0.884 | **595** |
| baseline | 0.693 | 0.895 | 0.883 | 0.883 | 591 |
| **delta_direct** | **0.765** | 0.918 | **0.924** | 0.885 | **843** |
| **channel_gate** | **0.767** | 0.918 | **0.922** | 0.884 | **823** |

**관찰**:
- Baseline: vanilla와 거의 동일 (모든 metric 차이 < 0.01)
- **delta_direct / channel_gate**: direction 88% → **92%**, Position R²(x) 0.69 → **0.77**, F_dir 595 → **843/823** (+40%)
- **Object accuracy는 모두 0.88로 유지** (semantic 정보 안 건드림)

## I.5 Dim-level 분석

### Vanilla top-50 direction dim이 각 모델에서 어떻게 변했나?

| Model | Vanilla top-50 중 여전히 top-50 (overlap) | 강화됨 (F↑) | 약화됨 (F↓) |
|---|---|---|---|
| baseline | **46/50** | 17/50 | 33/50 |
| delta_direct | 21/50 | **32/50** | 18/50 |
| channel_gate | 27/50 | **33/50** | 17/50 |

### 해석
- **Baseline**: vanilla의 top-50 중 46개가 여전히 top-50에 남아있다 = **거의 변화 없음**. 단순 LoRA 학습으로는 direction dim 구조 변화 미미.
- **delta_direct/channel_gate**: 
  - Top-50 overlap은 21-27로 줄었지만
  - **Vanilla top-50 중 2/3 (32, 33개)는 F-stat이 오히려 강해짐**
  - 즉 "기존 dim을 버린" 게 아니라 **"기존 direction dim을 유지 + 강화 + 일부 새 dim 추가"**

## I.6 Projector Weight는 얼마나 변했나?

Vanilla vs Trained의 mm_projector weight 직접 비교:

| Model | Layer 0 (Linear 1152→3584) | | Layer 2 (Linear 3584→3584) | |
|---|---|---|---|---|
|  | rel_diff | min row_cos | rel_diff | min row_cos |
| **baseline** | 1.3% | 0.999 | 1.6% | 0.999 |
| **delta_direct** | 4.7% | **0.986** | 5.8% | 0.992 |
| **channel_gate** | 4.8% | 0.990 | 5.9% | 0.992 |

- `rel_diff` = `||W_trained - W_vanilla|| / ||W_vanilla||`
- `row_cos` = 각 output dim을 만드는 weight row 벡터 간 cosine similarity

### 해석
- **Baseline: 1-2% 변화. projector는 거의 안 변함**
- **delta_direct/channel_gate: 5-6% 변화. 모든 row의 cosine > 0.98**
  - Weight의 **방향**은 거의 그대로 (cosine ≈ 1)
  - 크기와 bias만 미세하게 조정

### 왜 작은 변화가 큰 F-stat 증가를 만드나?
- Weight 방향이 같으면 → 각 dim이 인코딩하는 "의미"는 거의 동일
- 하지만 weight norm과 bias 미세 조정으로 → **4개 direction class의 distribution centroid가 살짝 이동**
- → Between-class variance 증가, within-class variance 유지
- → F-stat = (between / within) 40% 증가

직관: **"선명하게 분리되도록 살짝 밀어냄"**. 큰 re-architecture가 아니라 fine-tuning이 direction 축으로 class centroids를 push.

## I.7 결론

**Q9 ("fine-tuned가 뭘 바꿨나")의 답**:

1. **Baseline**: 단순 LoRA는 projector를 거의 안 바꿈. 성능 개선 미미.
2. **delta_direct / channel_gate**: 
   - Projector weight를 **5% 정도만 미세 조정**
   - 그 결과 **vanilla의 direction dim을 유지하면서 더 선명하게 분리**되도록 class centroid가 이동
   - F-stat이 40% 증가, direction accuracy 4%p 향상
3. **Semantic 정보는 그대로 유지** (object acc, dim separation 모두 유지)
4. → **explicit direction loss/module이 핵심**. 단순 LoRA로는 direction 특화가 안 된다.

---

# 7. 종합 발견 사항 (Summary of Findings)

| # | 발견 | 증거 |
|---|---|---|
| 1 | Mean-pool(N=729) 후에도 위치 정보는 살아있다 | Pre-proj R²(x)=0.73, R²(y)=0.90 on 8000 R2R_real_color |
| 2 | 위치 정보는 **low-rank subspace가 아닌 전체 D에 분산** 인코딩 | PCA top-50 (var 82%)으로 R²(x)=0.06에 불과 |
| 3 | Top-200 pos-correlated dim을 제거해도 position R²=0.56 (분산 인코딩 추가 증거) | Position-nulled ablation |
| 4 | Projector 통과 후 위치 정보 오히려 **강화** | Post-proj R²(x) 0.73→0.81 |
| 5 | Delta(두 프레임 차이)로 direction 91% 예측 | Pre-proj delta direction |
| 6 | Single frame, T-mean은 chance 근처 (시간 순서가 direction의 핵심) | Sanity checks |
| 7 | Position/Direction은 **같은 dim** (37/50), Object는 **분리된 dim** (3/50) | Top-50 dim overlap |
| 8 | Projector는 Position-Object 분리를 더 강화 (3/50 → 0/50) | Pre vs post overlap |
| 9 | **배경 복잡도가 direction encoding의 주적** (단색→실제장소: 91%→55%) | Testbed 4 conditions |
| 10 | F>5 direction dim 수: shape_color 665 → shape_place 99 (85% 감소) | F-stat distribution |
| 11 | 실제 영상(SSv2, KTH)에서도 direction encoding 작동 | SSv2 delta 67%, KTH single 73% |
| 12 | SSv2는 motion 필수 (single=chance), KTH는 자세만으로도 가능 | Single frame vs delta |
| 13 | LLM **Layer 1부터 즉시** last token이 vision 정보 흡수 | LLM hidden probe |
| 14 | Layer 4에서 position 정보 peak (R²=0.85) | LLM hidden probe |
| 15 | 깊은 layer일수록 raw 정보 감쇠, but 끝까지 chance 위 | LLM hidden probe |
| 16 | **Baseline (LoRA only)은 projector를 거의 안 바꿈** (1-2%) | Projector weight analysis |
| 17 | delta_direct/channel_gate는 projector를 **5% 미세 조정** (row cosine > 0.98) | Projector weight analysis |
| 18 | 5% 미세 조정만으로 direction F-stat **40% 증가** (595→843) | Phase 4 rigorous |
| 19 | delta_direct/channel_gate는 vanilla의 direction dim을 **유지 + 강화** (32/50 strengthened) | Phase 4 rigorous |
| 20 | Object accuracy는 모든 모델에서 유지 (semantic 건드리지 않음) | Phase 4 rigorous |

---

# 8. Method Design에의 시사점

## 8.1 표현 구조 (Representation Structure)
- **위치 정보가 distributed하게 인코딩**되어 있음 → 특정 dim을 강조하는 방식보다 **전체 D를 활용**하는 operation이 위치 정보 보존에 유리
- Position ≈ Direction, Object는 분리 → semantic을 건드리지 않고 **position/direction dim만 조작 가능**

## 8.2 배경 처리 (Background Handling)
- 배경이 direction encoding의 주적 → **background suppression** 또는 **foreground-focused attention**이 direction task에 효과적일 것
- F>5 dim 수가 85% 감소 → 배경이 direction signal을 "희석"

## 8.3 Projector 역할
- mm_projector tuning은 방향 특화에 매우 효율적: **5% weight 변화로 F-stat 40% 증가**
- 방법: explicit direction-related loss/module이 필요 (delta_direct, channel_gate)
- 단순 LoRA(baseline)는 projector를 거의 안 움직임 → direction 특화 학습이 안 일어남

## 8.4 LLM Supervision Layer
- Vision 정보가 Layer 1-4에 가장 풍부 → **얕은 layer에서 supervision**을 주는 것이 이론적으로 유리
- `llm_delta_direct`와 같이 LLM 중간 feature에서 loss를 거는 전략이 justified

## 8.5 Real-world Transfer
- 합성에서 관찰된 현상이 실제 영상에서도 재현됨 (SSv2, KTH)
- Synthetic-trained method가 real-world에 transfer할 근거

---

# 9. 부록: Implementation Details

## 9.1 Feature Extraction Pipeline
```python
frames = read_video_8frames(path)                     # list of (H, W, 3) numpy
pixel_values = processor.preprocess(frames)           # (8, 3, 384, 384) fp16

with torch.no_grad():
    out = encoder(pixel_values, output_hidden_states=True)
    pre = out.hidden_states[-1]                       # (8, 729, 1152) — pre-proj patch tokens
    post = projector(pre)                             # (8, 729, 3584) — post-proj patch tokens
    
    pre_mean = pre.mean(dim=1)                        # (8, 1152) — pre-proj mean-pooled
    post_mean = post.mean(dim=1)                      # (8, 3584) — post-proj mean-pooled
```

**주의**: post-proj는 **반드시 per-patch에 projector 적용 후 mean-pool** (아니면 GELU 비선형성 때문에 부정확).

## 9.2 Probe Implementation

### Ridge Regression (GPU, closed-form)
```python
# w = (X'X + αI)^{-1} X'y, bias 포함
X_b = torch.cat([X, ones], dim=1)
reg = α * torch.eye(D+1, device=device)
reg[-1, -1] = 0  # don't regularize bias
w = torch.linalg.solve(X_b.T @ X_b + reg, X_b.T @ y)
r2 = 1 - ss_res / ss_tot
```

### Classification (GPU, Adam)
```python
model = nn.Linear(D, n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
for epoch in range(300):
    loss = F.cross_entropy(model(X_train), y_train)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
acc = (model(X_test).argmax(1) == y_test).float().mean()
```

### Cross-validation
- 5-fold (KFold for regression, StratifiedKFold for classification)
- `random_state=42` 고정
- StandardScaler를 fold마다 새로 fit (train에 fit, test에 transform)

## 9.3 Top-k Dim Selection Criteria

| Attribute | Score per dim |
|---|---|
| Position | `sqrt(corr(X[:,d], pos_x)² + corr(X[:,d], pos_y)²)` |
| Direction | one-way ANOVA F-stat across 4 direction groups on **delta** vector |
| Object | one-way ANOVA F-stat across 20 object groups on **mid frame** |

선정: `np.argsort(-score)[:k]`

## 9.4 LLM Hidden State Extraction
```python
# CRITICAL: use tokenizer_image_token (not plain tokenizer)
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

input_ids = tokenizer_image_token(
    prompt_with_image_token, tokenizer, IMAGE_TOKEN_INDEX,
    return_tensors="pt"
).unsqueeze(0).to(device)

# Merge vision tokens into input embeddings
with torch.no_grad():
    (_, _, attn_mask, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None,
        images=[video_tensor], modalities=["video"]
    )
    outputs = model.forward(
        input_ids=None, inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        output_hidden_states=True, return_dict=True
    )

# outputs.hidden_states: tuple of (n_layers+1,) — [0]=embedding, [1..n_layers]=each transformer layer output
last_token_hidden_at_layer_k = outputs.hidden_states[k][0, -1, :].detach().cpu().numpy()
```

## 9.5 Trained Projector Loading
```python
# Trained projector weights are in non_lora_trainables.bin (not adapter_model.safetensors)
state = torch.load(os.path.join(model_dir, "non_lora_trainables.bin"), map_location="cpu")
proj_state = {
    k.split("mm_projector.")[-1]: v 
    for k, v in state.items() if "mm_projector" in k
}
projector.load_state_dict(proj_state)
```

## 9.6 Multi-GPU Feature Extraction (for R2R_real_color 8000 videos)
- 6 GPU에 1333개씩 split
- `torch.multiprocessing.spawn`으로 병렬 worker
- 각 worker가 SigLip + 4 projectors를 로드 후 batch 처리
- 병합 시 chunk 파일 concat → single npz

## 9.7 Compute Resources
| Phase | GPU 사용량 | Runtime |
|---|---|---|
| Feature extraction (8000 vids, 6 GPU parallel) | ~4 GB per GPU | ~5 min |
| Linear probes (GPU 1개) | ~2-5 GB | ~5-10 min per phase |
| LLM probing (7B fp16, 400 vids) | ~16 GB | ~15 min |
| Projector weight analysis | CPU-only | ~10 sec |

---

# 10. 파일 구조

```
analysis/token_linear_probing/
├── FINAL_REPORT.md                  # ← THIS DOCUMENT (통합 최종 보고서)
├── README.md                        # 이전 요약 (Legacy, FINAL_REPORT가 최신)
│
├── extract_features.py              # Testbed feature extraction
├── extract_r2r_real_color.py        # R2R_real_color extraction (6-GPU parallel)
├── probe_gpu.py                     # 메인 GPU probe (vanilla)
├── dim_analysis_r2r_real_color.py   # Dim overlap analysis (vanilla)
│
├── features/
│   ├── R2R_*.npz                    # Testbed mean-pooled features
│   ├── R2R_real_color.npz           # 메인 데이터 (pre_mean + post_mean, vanilla)
│   └── R2R_real_color_v2_multiproj.npz  # 4개 projector 각각 적용 후 (Phase 4 rigorous용)
│
├── 2026-04-12_overnight/
│   ├── PLAN.md, PLAN_trained_models.md    # 실험 계획
│   ├── IMPL_DETAILS.md                    # 구현 세부사항
│   ├── REPORT_archive.md                  # 초기 REPORT (deprecated)
│   ├── projector_weight_analysis.md       # Projector weight 변화 분석
│   │
│   ├── exp1_subspace.py                   # Part B (PCA, null ablation)
│   ├── exp2_complexity.py                 # Part F (complexity gradient)
│   ├── exp3_llm_probing.py                # Part H (LLM hidden probe)
│   ├── exp4_v2_rigorous.py                # Part I (trained models, rigorous)
│   ├── exp5_realworld.py                  # Part G (SSv2, KTH)
│   ├── extract_v2_multi_projector.py      # 4 projector 재추출
│   │
│   ├── exp1_subspace_results.json         # Part B results
│   ├── exp2_complexity_results.json       # Part F results
│   ├── exp3_llm_probing_results.json      # Part H results
│   ├── exp4_v2_rigorous_results.json      # Part I results (final)
│   ├── exp5_realworld_results.json        # Part G results
│   │
│   └── fstats_v2_{vanilla,baseline,delta_direct,channel_gate}.npz
│        # 각 모델의 per-dim F-stat (direction, object)
│
└── results/                         # 추가 JSON/npz 결과
```

---

# 맺음말

이 프로젝트는 "왜 VLM이 direction을 잘 못하는가?"라는 질문에서 출발해서, vision token representation을 **dim-level로 해부**했다. 

핵심적으로 밝혀낸 것:
- **위치 정보는 전체 D에 분산되어 있고, mean-pooling 후에도 살아있다**
- **Position/Direction은 같은 dim, Object는 분리된 dim 사용**
- **배경 복잡도가 direction encoding의 주적**
- **LLM은 Layer 1부터 vision 정보를 흡수, Layer 4가 peak**
- **Fine-tuned 방법(delta_direct, channel_gate)은 projector를 5% 미세 조정해서 direction dim을 선명하게 분리시킴**

이 분석은 방법론적으로 "왜 우리 모듈들이 작동하는가"를 representation level에서 뒷받침한다. 앞으로의 method design (특히 background suppression, shallow-layer LLM supervision)에 직접적인 방향을 제시한다.
