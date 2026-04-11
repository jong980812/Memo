# Token Linear Probing: Position & Semantic Information in Vision Tokens

## Research Question
Vanilla LLaVA-Video-Qwen2의 SigLip vision encoder에서 나오는 vision tokens (projection 전/후)에:
1. 물체의 **위치 정보**가 mean-pooling(N축 평균) 후에도 살아있는가?
2. **어떤 dimension**들이 위치 정보 vs semantic 정보를 인코딩하는가?
3. **Projector** 통과 후에도 이 경향이 유지되는가?
4. 시간축으로 쌓은 **(T, D) 벡터**에서 움직임 방향을 예측할 수 있는가?
5. T축까지 평균내면 어떻게 되는가?

## Background
- SigLip ViT: absolute position embedding → patch embedding + pos_emb → 27 layers self-attention
- Output: (B, 729, 1152) — 729 = 27x27 patches, 1152-dim
- Projector: mlp2x_gelu → (B, 729, 3584)
- Position embedding이 translational equivariance를 깨뜨림 → 같은 물체가 다른 위치에 있으면 모든 패치 값이 달라짐
- **핵심 가설**: mean-pool(N=729) → D-dim vector에 위치 정보가 남아있다

## Model
- SigLip-SO400M-patch14-384 (from LLaVA-Video-7B-Qwen2)
- Projector: Linear(1152→3584) → GELU → Linear(3584→3584)
- Weights: `/data/dataset/LLaVA-Video-100K-Subset/models--lmms-lab--LLaVA-Video-7B-Qwen2/`

## Data

### Testbed (small scale)
- `/local_datasets/vlm_direction/vlm_direction_testbed/`
- R2R_obj_place: 200 videos, 합성 도형/실제 물체, 32 frames, 384x384
- 4방향 (up/down/left/right) x 50개씩

### R2R_real_color (large scale)
- `/local_datasets/vlm_direction/R2R_real_color/`
- **8000 videos**, 20 COCO objects x 4 directions x 100 each
- 8 frames, 384x384, 단색 배경, 직선 운동
- Random start/end positions

---

## Results 1: Testbed (200 videos, 8 frames)

| Metric | Pre-proj (1152-d) | Post-proj (3584-d) | Chance |
|--------|-------------------|--------------------|--------|
| **Position R2 (x)** | **0.925** | **0.914** | ~0 |
| **Position R2 (y)** | **0.942** | **0.942** | ~0 |
| Direction (single frame) | 0.180 | 0.185 | 0.25 |
| **Delta direction (last-first)** | **0.735** | **0.680** | 0.25 |
| Object class (26 cls) | 0.165 | 0.200 | 0.038 |

**결론**: N축 평균내도 위치 R2 > 0.92. Delta로 방향 73%. 단 200 samples로 temporal stack은 overfitting.

---

## Results 2: R2R_real_color (8000 videos, 20 objects, 8 frames, GPU probe)

| Metric | Pre-proj (1152-d) | Post-proj (3584-d) | Chance |
|--------|-------------------|--------------------|--------|
| **Position R2 (x)** | 0.727 | **0.806** | ~0 |
| **Position R2 (y)** | 0.895 | **0.933** | ~0 |
| Direction (single frame) | 0.408 | 0.370 | 0.25 |
| **Delta direction (last-first)** | **0.912** | **0.882** | 0.25 |
| Direction T-mean | 0.358 | 0.331 | 0.25 |
| **8-frame stack direction** | **0.820** | **0.800** | 0.25 |
| **Object class (single frame, 20 cls)** | **0.927** | 0.883 | 0.05 |
| **Object class T-mean** | **0.946** | **0.909** | 0.05 |

### Key Observations
- **Position R2**: post-proj가 pre-proj보다 높음 (0.73→0.81) — projector가 위치 정보를 정리
- **Delta direction 91%**: mean-pooled 벡터 2개 차이로 방향 거의 완벽 예측
- **8-frame stack 82%**: 8000 samples면 overfitting 없이 작동 (이전 200 samples에서는 chance)
- **Object class 93%**: semantic 정보 매우 강하게 존재
- **T-mean**: object 올라가고 (93→95%) direction 내려감 (91→36%) — 시간 평균 = semantic 강화, 방향 소실

---

## Results 3: Dimension Analysis (R2R_real_color, 8000 videos)

### Top-50 dim overlap

| Overlap | Pre-proj (1152-d) | Post-proj (3584-d) | Random expectation |
|---------|-------------------|--------------------|-------------------|
| **Position ∩ Direction** | **37/50** | **30/50** | 2.2 / 0.7 |
| **Position ∩ Object** | **3/50** | **0/50** | 2.2 / 0.7 |
| **Direction ∩ Object** | **2/50** | **1/50** | 2.2 / 0.7 |

### Interpretation
1. **Position과 Direction은 거의 같은 dim 사용** (37/50, 30/50 overlap)
   - direction = position 변화이므로, position dim의 delta가 곧 direction
2. **Object(semantic)는 완전히 다른 dim 사용** (3/50, 0/50 overlap)
   - **위치 정보와 semantic 정보가 D 벡터 내에서 분리된 차원에 인코딩**
3. **Projector 통과 후 더 깔끔하게 분리** (0/50 overlap)
   - MLP projector가 position/semantic 정보를 더 분리해서 LLM에 전달

### Cross-probe (top-50 dims만 사용)

| Probe | Pre-proj | Post-proj | Chance |
|-------|----------|-----------|--------|
| Object from position dims | 0.704 | 0.749 | 0.05 |
| Object from object dims | 0.767 | 0.759 | 0.05 |
| Direction from direction dims | 0.757 | 0.643 | 0.25 |
| Direction from position dims | 0.746 | 0.631 | 0.25 |

- Top-50 dim이 겹치지 않아도, 50개 dim에 약한 신호가 섞여있어서 cross-probe 성능이 나옴
- 핵심은 **가장 강한 신호를 담는 dim이 분리**되어 있다는 것

---

## Summary of Findings

1. **N축(패치) 평균내도 위치 정보 보존**: R2 = 0.73~0.93 (linear probe)
2. **Projector 통과 후 위치 정보 오히려 강화**: pre(0.73)→post(0.81) for x
3. **Delta(두 프레임 차이)로 방향 91% 예측**: (T,D) 시퀀스에 움직임 정보 존재
4. **8-frame stack으로 방향 82% 예측**: 충분한 데이터(8000)면 overfitting 해결
5. **Object class 93%**: mean-pooled 벡터에 semantic 정보도 강하게 존재
6. **T축 평균내면 object↑(95%) direction↓(36%)**: 시간 평균 = semantic 강화, 방향 소실
7. **위치/방향과 semantic이 서로 다른 dim에 인코딩**: overlap ≈ 0/50
8. **Projector가 이 분리를 더 강화**: post-proj에서 position-object overlap = 0/50

## Files
- `extract_features.py` — Feature extraction (testbed)
- `extract_r2r_real_color.py` — Feature extraction (R2R_real_color, 6-GPU parallel)
- `quick_probe.py` / `quick_probe_8frames.py` — Quick probing scripts
- `probe_r2r_real_color.py` — Full probe on R2R_real_color
- `dim_analysis_r2r_real_color.py` — Dimension overlap analysis
- `features/` — Extracted .npz files
- `results/` — JSON results + dim analysis .npz
- `figures/` — Plots

