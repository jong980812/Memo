# Overnight Experiments Plan (2026-04-12)

## Research Questions
1. 위치 정보가 특정 부분공간(subspace)에 있는가, 전체 D에 퍼져있는가?
2. 시각적 복잡도(도형→사물, 단색→장소)가 증가하면 direction dim이 약해지는가?
3. LLM을 거치면서 text token(last token 등)이 position/direction 정보를 실제로 가져가는가?

## Constraints
- GPU memory < 20GB per experiment
- Training running on GPUs (use 1 GPU, be careful)
- ~6 hours runtime budget

---

## Phase 1: Position Subspace Analysis
**Q: 위치 정보가 low-rank subspace에 있는가?**

### Exp 1.1: PCA dimensionality of position encoding
- R2R_real_color features (8000 videos)
- PCA on per-frame mean-pooled tokens
- Plot: position R2 vs number of PCA dims (1, 2, 5, 10, 20, 50, 100, 200, 500, 1152)
- If R2 plateaus at k << D → position is in a k-dim subspace

### Exp 1.2: Top-k position dims vs random dims
- Take top-k dims by position correlation
- Compare position R2 using: top-k pos dims, top-k random dims, top-k object dims
- Shows whether position info is concentrated in specific dims or spread

### Exp 1.3: Position-nulled representation
- Project out top-k position dims from the D-dim vector
- Can we still classify objects? (should be yes if position/semantic are separated)
- Can we still predict direction from delta? (should drop significantly)

---

## Phase 2: Complexity Gradient
**Q: 시각적 복잡도가 증가하면 direction encoding이 약해지는가?**

### Exp 2.1: Compare across existing testbed conditions
Already have features for 4 conditions (R2R): shape_color, shape_place, obj_color, obj_place
- Compare: position R2, delta direction acc, object F-stats
- Hypothesis: place backgrounds and real objects → noisier direction dims

### Exp 2.2: Create and test real_obj + real_place videos
- Use R2R_real_color's COCO objects + place365 backgrounds
- Generate ~2000 videos (20 objects x 4 dirs x 25 each)
- Compare with R2R_real_color (solid bg) to isolate background effect

### Exp 2.3: Direction dim F-stat distribution across conditions
- Plot F-stat histograms for direction across all conditions
- Show whether direction signal concentrates in fewer dims or spreads out

---

## Phase 3: LLM Hidden State Probing
**Q: LLM의 text token이 vision token의 position/direction 정보를 가져가는가?**

### Exp 3.1: Extract LLM hidden states
- Load LLaVA-Video-7B-Qwen2 (fp16, ~14GB)
- Feed subset of R2R_real_color (200 videos for memory)
- Simple prompt: "Describe the video."
- Extract hidden states at: vision token positions, last token, per-layer

### Exp 3.2: Probe LLM hidden states
- At each layer: probe last token for position, direction, object
- Compare: layer 0 (just after embedding) vs middle vs final layer
- If last token has direction info → LLM successfully extracts it from vision tokens

---

## Phase 4-5: Trained Model Comparison (NEW)
**Q: Baseline/delta-direct/channel-gate가 direction을 더 잘하는 이유는?**

### Exp 4: Projector가 바뀌면 dim 구조가 어떻게 변하나?
- 4combo_new/work_dirs의 3개 모델 (baseline, delta_direct, channel_gate)
- mm_mlp_adapter가 학습됨 → post-proj features가 달라짐
- 각 모델의 trained projector를 로딩해서 R2R_real_color features에 적용
- Direction F-stat 비교: 어떤 dim이 강화/약화됐는지
- Top-50 direction dim overlap: vanilla vs 각 model

### Exp 5b: Real-world Datasets (SSv2, KTH)
- SSv2: left/right pushing (500 videos)
- KTH: left/right walking/jogging (899 videos)
- 실제 영상에서도 direction encoding이 작동하는지
- Synthetic과 비교

## Execution Order
1. Phase 1 ✅ (완료)
2. Phase 2 ✅ (완료)
3. Phase 4-5: Trained model comparison (training 끝나면 자동 실행)
4. Phase 5b: Real-world datasets (training 끝나면 자동 실행)
5. Phase 3: LLM probing (마지막, GPU 많이 씀)

## Runner
`run_all_overnight.sh` — training 끝나면 자동으로 Phase 4→5b→3 순서 실행 (PID: 135872)

## Output
- All results in `2026-04-12_overnight/`
- Each experiment: script + results JSON
- Summary: REPORT.md
