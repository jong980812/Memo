# Overnight Experiment Report (2026-04-12)

## Summary

**3가지 핵심 질문에 대한 답:**
1. **위치 정보가 특정 subspace에 있는가?** → **No.** 전체 D=1152에 분산 인코딩.
2. **시각적 복잡도가 direction encoding을 약화시키는가?** → **Yes.** 특히 배경 복잡도가 주적.
3. **LLM의 last token이 vision 정보를 가져가는가?** → **Yes.** Layer 1부터 즉시, Layer 4에서 peak.

**Bonus 발견:**
4. **학습된 모델 (baseline/delta_direct/channel_gate)은 dim 구조를 재구성**: linear probe accuracy는 비슷하지만 dim 사용이 완전히 다름.
5. **Real-world videos (SSv2, KTH)에서도 direction encoding 작동**.

---

## Phase 1: Position Subspace Analysis ✅

### Exp 1.1: PCA dimensionality
**위치 정보는 low-rank subspace에 있지 않다.**

| PCA dims | Var explained | R2(x) | R2(y) |
|---|---|---|---|
| 50 | 81.7% | 0.060 | 0.493 |
| 200 | 91.6% | 0.285 | 0.782 |
| 500 | 97.1% | 0.522 | 0.843 |
| 1000 | 99.7% | 0.633 | 0.870 |
| 1152 | 100% | **0.727** | **0.895** |

→ Top PCA 50 (분산 82%)으로는 R2(x)=0.06. 거의 모든 dim 필요.

### Exp 1.3: Position-nulled ablation
**Top-200 position dim 제거해도 거의 영향 없음 (위치 정보가 분산 인코딩되어 있다는 증거).**

| 제거한 pos dims | Object | Direction | Position R2(x) |
|---|---|---|---|
| 0 | 0.927 | 0.912 | 0.73 |
| 50 | 0.927 | 0.900 | 0.60 |
| 100 | 0.926 | 0.897 | 0.59 |
| 200 | **0.924** | **0.886** | **0.56** |

→ Object는 전혀 영향 없음 (position/semantic 분리 확인).
→ Position R2가 0.56까지만 떨어짐 (나머지 952 dim에 위치 정보 잔존).

---

## Phase 2: Complexity Gradient ✅

### Exp 2.1: Conditions (200 vids each)
**배경 복잡도가 direction encoding의 주적.**

| 조건 | 배경 | 물체 | R2(x) | Delta dir | F_dir top50 |
|---|---|---|---|---|---|
| shape_color | 단색 | 도형 | **0.986** | **0.910** | 38.9 |
| obj_color | 단색 | 실제물체 | 0.889 | 0.885 | 18.4 |
| shape_place | **실제배경** | 도형 | 0.930 | **0.555** | 12.5 |
| obj_place | **실제배경** | 실제물체 | 0.925 | 0.700 | 13.1 |

### Exp 2.3: F-stat distribution
**실제 배경 추가 시 direction-strong dim 수가 85% 감소** (665→99).

| 조건 | dims F>5 |
|---|---|
| shape_color | 665 |
| obj_color | 337 |
| shape_place | **99** |
| obj_place | 116 |

---

## Phase 3: LLM Hidden State Probing ✅

**Last token의 layer별 정보 (400 R2R_real_color videos):**

| Layer | Direction | Object | Pos R2(x) |
|---|---|---|---|
| 0 (embed) | 0.250 (chance) | 0.085 | -0.005 |
| **1** | **0.525** | **0.663** | 0.620 |
| **4** | **0.573** ★ | 0.645 | **0.846** ★ |
| 8 | 0.480 | 0.603 | 0.839 |
| 12 | 0.443 | 0.485 | 0.666 |
| 16 | 0.483 | 0.498 | 0.609 |
| 20 | 0.463 | 0.483 | 0.598 |
| 24 | 0.463 | 0.478 | 0.586 |
| 28 (last) | 0.468 | 0.530 | 0.498 |

**해석:**
- Layer 1부터 last token이 vision 정보 흡수 (chance → 52% direction)
- **Layer 4가 peak** (position R2 0.85, direction 57%)
- 깊어질수록 raw 정보 감쇠 → LLM이 task-specific representation으로 변환
- 끝까지 chance보다 높음 (vision 정보가 최종 출력까지 살아있음)

---

## Phase 4 RIGOROUS: Trained Model Comparison ✅

**Vanilla vs Baseline vs Delta-direct vs Channel-gate (R2R_real_color, 8000 videos)**

### Per-patch projection, then mean-pool (correct, no approximation)

| Model | Pos R2(x) | Pos R2(y) | Delta dir | Obj acc | F_dir mean | F_dir top50 | Overlap w/ vanilla | Vanilla top-50 strengthened |
|---|---|---|---|---|---|---|---|---|
| vanilla | 0.691 | 0.895 | 0.884 | 0.884 | 107 | **595** | - | - |
| baseline | 0.693 | 0.895 | 0.883 | 0.883 | 106 | 591 | **46/50** | 17/50 |
| **delta_direct** | **0.765** | 0.918 | **0.924** | 0.885 | **151** | **843** | 21/50 | **32/50** |
| **channel_gate** | **0.767** | 0.918 | **0.922** | 0.884 | **150** | **823** | 27/50 | **33/50** |

### 핵심 발견 (RIGOROUS — 최종)
1. **Linear probe accuracy**:
   - Direction: vanilla 0.88 → delta_direct/channel_gate **0.92** (+4%p)
   - Position R2(x): vanilla 0.69 → ours **0.77** (+0.08)
   - Object acc: 모두 ≈ 0.88 (유지)
2. **Dimension structure**:
   - **Baseline**: vanilla와 거의 동일한 dim 사용 (46/50 overlap). 학습 효과 제한적.
   - **delta_direct/channel_gate**: vanilla의 top-50 direction dim 중 **2/3를 강화** (32/50, 33/50)하면서 일부 새 dim (21-27/50 overlap)도 활용.
3. **F-stat 40%+ 증가**: direction 신호 강도 595 → 843/823.
4. **Semantic과 분리 유지**: dir-obj overlap 모두 ≤ 1/50.

### 이전 Approximation 결과와의 차이 (왜 달라졌나)

| Model | Approximation | **Rigorous (correct)** | 해석 변화 |
|---|---|---|---|
| baseline overlap | 34/50 | **46/50** | 더 비슷 |
| delta_direct overlap | 14/50 | **21/50** | 더 비슷 |
| delta_direct strengthened | 10/50 | **32/50** | **반대 결론** |
| channel_gate strengthened | 13/50 | **33/50** | **반대 결론** |

**이전 (틀린) 해석**: "delta_direct가 vanilla의 direction dim을 버리고 새 dim을 만든다"
**RIGOROUS (맞는) 해석**: "delta_direct/channel_gate는 **vanilla의 direction dim을 유지/강화**하면서 일부 새 dim도 추가"

### Method 정당성 해석
- delta_direct/channel_gate가 direction을 잘 하는 이유:
  1. **기존 direction dim을 강하게 만듦** (F-stat 40% 증가)
  2. **Semantic dim은 건드리지 않음** (object acc 유지, dir-obj overlap 유지)
  3. **Position R2도 올림** (0.69→0.77) — projector가 위치 정보 encoding을 더 정제함
- Baseline은 단순 LoRA 훈련만으로는 direction dim을 크게 바꾸지 못함. **explicit direction-related loss/module이 핵심**.

---

## Phase 5b: Real-world Datasets ✅

| Dataset | Stage | Delta dir | Single | T-mean | F>5 dims |
|---|---|---|---|---|---|
| **SSv2** (push left/right) | pre | 0.662 | 0.452 | 0.440 | 49 |
|  | post | 0.666 | 0.448 | 0.466 | **441** |
| **KTH** (walk left/right) | pre | 0.699 | **0.733** | 0.666 | 390 |
|  | post | 0.655 | 0.687 | 0.670 | **1400** |

**해석:**
- **SSv2**: chance(50%) 위로 67% — push 방향이 motion에 의존 → single frame은 chance, delta가 잘 잡음
- **KTH**: single frame만으로 73% — 사람 걷는 방향은 frame에 그대로 보임
- **Post-proj에서 F>5 dim 수 폭증** (49→441, 390→1400) — projector가 실제 영상에서 direction 신호를 여러 dim에 확산

---

## Implications for Method Design

1. **위치 정보가 분산 인코딩** → 특정 dim만 강조하는 attention보다 전체 D를 활용하는 게 유리할 수 있음
2. **Position과 Semantic이 분리된 dim** → semantic 정보를 손상시키지 않고 position dim만 조작 가능
3. **배경이 noise source** → background suppression이 direction task에 도움 될 것
4. **Trained methods가 dim을 재구성** → projector tuning이 핵심. delta-direct/channel-gate처럼 direction signal을 더 적은 dim에 집중시키는 게 유효한 전략
5. **LLM은 vision 정보를 layer 1-4에서 가장 잘 보존** → 깊은 layer에서는 추상화로 변환되므로, 깊은 layer에서 추가 supervision을 주는 게 효과적일 수 있음 (llm_delta_direct 같은 방법)

---

## Files
- `exp1_subspace.py` — Phase 1
- `exp2_complexity.py` — Phase 2
- `exp4_trained_models.py` — Phase 4-5 (trained model comparison)
- `exp5_realworld.py` — Phase 5b (SSv2, KTH)
- `exp3_llm_probing.py` — Phase 3 (LLM hidden states)
- `*_results.json` — All numerical results
- `fstats_*.npz` — Per-model F-stat arrays
- `overnight.log` — Full execution log
