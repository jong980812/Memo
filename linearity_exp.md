# VLM Vision Feature 분석 요약

## 모델: LLaVA-Video-7B-Qwen2
- Vision encoder: SigLIP-SO400M-patch14-384
- Projector: mlp2x_gelu (Linear 1152→3584, GELU, Linear 3584→3584)
- LLM: Qwen2-7B

## 데이터셋
- **E2E_VP** (합성): 단순 도형이 4방향(down/left/right/up)으로 이동하는 영상
- **E2E_real_VP** (실제): 실제 물체가 4방향으로 이동하는 영상

---

## 1. 시간 축 선형성 분석 (Temporal Linearity)

### 방법
- MLP projector 전후의 프레임별 feature 추출
- 패치 축 mean pool → 프레임당 벡터 1개 → (T, D), T=8프레임
- 각 dimension별 선형 회귀 (프레임 인덱스 vs 값) → R²로 선형성 측정
- PCA → PC1 설명력으로 전체 궤적의 선형성 측정

### 결과 — Projector 통과 후 (3584 dims)

| 방향 | 데이터셋 | R² 평균 | R²>0.9 | R²>0.8 | R²>0.7 | PC1 |
|------|----------|---------|--------|--------|--------|-----|
| down | 합성 | 0.266 | 54 (1.5%) | 194 (5.4%) | 346 (9.7%) | 69.3% |
| left | 합성 | 0.164 | 10 (0.3%) | 35 (1.0%) | 90 (2.5%) | 74.7% |
| right | 합성 | 0.158 | 5 (0.1%) | 24 (0.7%) | 88 (2.5%) | 74.4% |
| up | 합성 | 0.226 | 24 (0.7%) | 120 (3.3%) | 219 (6.1%) | 69.2% |
| down | 실제 | 0.317 | 42 (1.2%) | 168 (4.7%) | 374 (10.4%) | 44.8% |
| left | 실제 | 0.119 | 1 (0.0%) | 8 (0.2%) | 21 (0.6%) | 41.6% |
| right | 실제 | 0.141 | 1 (0.0%) | 11 (0.3%) | 33 (0.9%) | 41.4% |
| up | 실제 | 0.287 | 34 (0.9%) | 150 (4.2%) | 328 (9.2%) | 45.6% |

### 결과 — Projector 통과 전 (1152 dims)

| 방향 | 데이터셋 | R² 평균 | R²>0.9 | PC1 |
|------|----------|---------|--------|-----|
| down | 합성 | 0.318 | 25 (2.2%) | 61.6% |
| left | 합성 | 0.198 | 5 (0.4%) | 66.8% |
| right | 합성 | 0.185 | 4 (0.3%) | 64.6% |
| up | 합성 | 0.282 | 23 (2.0%) | 58.7% |
| down | 실제 | 0.270 | 6 (0.5%) | 38.4% |
| left | 실제 | 0.114 | 1 (0.1%) | 45.1% |
| right | 실제 | 0.111 | 1 (0.1%) | 41.1% |
| up | 실제 | 0.254 | 11 (1.0%) | 37.7% |

### 핵심 발견
1. **대부분의 dim은 비선형** (R² 평균 ~0.2)이지만, **소수(1~10%)는 R²>0.7로 선형**
2. **down/up이 left/right보다 선형 dim이 더 많음** — SigLIP의 수직 위치 인코딩이 더 강할 가능성
3. **합성이 실제보다 PC1이 높음** (~70% vs ~44%) — 단순 배경이라 궤적이 더 깔끔
4. **Projector가 선형성을 크게 바꾸지 않음** — 전후 R² 분포가 유사

---

## 2. Linear Probe: 방향 분류

### 방법
- 각 영상: after_proj feature 추출, 패치 mean pool → (T, D)
- 각 dim별 기울기(slope) 계산 (프레임 인덱스 대비 선형 회귀) → slope 벡터 (D,)를 영상 표현으로 사용
- frame_mean (시간 평균)도 비교 대상으로 테스트
- Logistic regression, 5-fold stratified CV
- 비교: 전체 dim vs R²>0.7 dim vs 랜덤 subset

### 결과 — 합성 (E2E_VP, 80개 영상 = 클래스당 20개)

영상 80개 평균 기준 R²>0.7인 dim: **13개**, R²>0.8 = 0개, R²>0.9 = 0개

| 표현 | Dim 수 | 정확도 |
|------|--------|--------|
| slope (전체) | 3584 | **1.000 ± 0.000** |
| slope (R²>0.7) | 13 | **1.000 ± 0.000** |
| slope (랜덤 13개) | 13 | 0.938 ± 0.040 |
| frame_mean (전체) | 3584 | 0.650 ± 0.192 |
| frame_mean (R²>0.7) | 13 | 0.625 ± 0.079 |
| frame_mean (랜덤 13개) | 13 | 0.600 ± 0.075 |

### 결과 — 실제 (E2E_real_VP, 80개 영상 = 클래스당 20개)

영상 80개 평균 기준 R²>0.7인 dim: **0개** (다양한 실제 영상에서 일관되게 선형인 dim 없음)

| 표현 | Dim 수 | 정확도 |
|------|--------|--------|
| slope (전체) | 3584 | **1.000 ± 0.000** |
| frame_mean (전체) | 3584 | 0.013 ± 0.025 |

### 핵심 발견
1. **Slope 표현은 합성/실제 모두 100%** — 전체 dim 사용 시
2. **합성에서 13개 선형 dim만으로 100%** (랜덤 13개는 93.8%) — 선형 dim이 핵심 정보를 담고 있음
3. **frame_mean은 실제에서 1.3%** (찍는 것과 동일) — 정적 표현에는 방향 정보 없음
4. **실제 영상에는 전역적으로 선형인 dim이 0개**지만, slope는 여전히 완벽 작동 — 각 영상 내에서는 선형이지만 영상마다 다른 dim이 선형인 것
5. 13개 선형 dim은 단순 노이즈가 아님 — 랜덤 13개(93.8%) vs 선형 13개(100%)

---

## 3. Mean-Pooled Delta가 방향을 분리하는 이유

`mean_pool(frame[i+1] - frame[i])`의 cosine similarity가 반대 방향 간 -0.99가 나온다는 것은
**SigLIP의 positional embedding이 패치 feature에 공간 위치 정보를 인코딩**하고 있다는 증거.

mean pool 후에도 방향 편향이 남는 이유:
- 물체가 떠나는 윗쪽 패치: feature ≈ -(content + pos_embed_위)
- 물체가 도착하는 아랫쪽 패치: feature ≈ +(content + pos_embed_아래)
- pos_embed_위 ≠ pos_embed_아래 이므로 상쇄되지 않고 방향 정보가 남음

---

## 파일 구조
```
results/
├── temporal_linearity/
│   ├── E2E_VP/              # 합성 데이터셋
│   │   ├── LLaVA-Video-Qwen2_{before,after}_proj_{방향}_*.png
│   │   └── LLaVA-Video-Qwen2_{방향}_{영상}_data.npz
│   └── E2E_real_VP/         # 실제 데이터셋
│       ├── LLaVA-Video-Qwen2_{before,after}_proj_{방향}_*.png
│       └── LLaVA-Video-Qwen2_{방향}_{영상}_data.npz
├── linear_probe/
│   ├── E2E_VP_default_linear_probe.json
│   └── E2E_real_VP_linear_probe.json
├── delta_pool/              # Delta mean/max pool 비교
├── patch_position/          # 패치별 위치 분석 (base 모델)
├── patch_position_lora/     # LoRA 모델
├── patch_position_lora_v4/  # Motion query v4 모델
└── SUMMARY.md               # 이 파일
```

