# Trained Model Dimension Analysis Plan

## Motivation
Vanilla LLaVA-Video는 direction을 잘 못하는데, baseline(LoRA fine-tuned)이나 
delta-direct, channel-gate는 direction을 잘한다. **왜?**

가설: 학습 과정에서 direction을 인코딩하는 dim이 강화되거나, 새로운 dim이 활성화됨.
이를 dimension-level로 분석해서 each method가 뭘 다르게 하는지 밝힌다.

## Models to Compare
1. **Vanilla** (no fine-tuning): 이미 분석 완료
2. **Baseline v5** (LoRA, shape_simple_v5): `4combo_v5/work_dirs/llava-video-7b-qwen2_baseline_shape_simple_v5_lora-r64_f8_ep1_lr1e-5/`
3. **Delta-direct v5**: `4combo_v5/work_dirs/llava-video-7b-qwen2_delta_direct_shape_simple_v5_lora-r64_f8_ep1_lr1e-5/` (학습 중)
4. **Channel-gate v3**: `4combo_exp/work_dirs/llava-video-7b-qwen2_channel_gate_synv3_lora-r64_f8_ep1_lr1e-5/`

## Analysis Plan

### Phase 4: Trained Model Feature Extraction
- Load each model (LoRA + base)
- Extract vision tokens (same as vanilla: SigLip + projector)
  - NOTE: LoRA는 LLM만 바꾸므로 vision encoder + projector는 동일할 수 있음!
  - 확인 필요: projector weights가 학습됐는지 (mm_tunable_parts에 mm_mlp_adapter 포함)
- 만약 projector가 바뀌었으면: pre-proj는 동일, post-proj가 달라짐

### Phase 5: Dimension Comparison
- 각 모델의 post-proj features로:
  - Position R2, Delta direction, Object class 비교
  - Direction F-stat distribution 비교
  - Top-50 direction dims overlap: vanilla vs baseline vs delta-direct vs channel-gate
  - 새로 활성화된 direction dims 찾기

### Phase 6: LLM Hidden State Comparison
- 각 모델의 LLM hidden states에서:
  - Last token의 direction/position/object probe (layer별)
  - Vanilla vs trained: 어떤 layer에서 direction 정보가 더 강해지는지
  - Method별 차이: delta-direct는 explicit direction loss가 있으니 다른 패턴?

## Key Question
- Baseline이 direction을 잘하는 이유: projector가 방향 dim을 강화? LLM이 더 잘 읽어감?
- Delta-direct가 추가로 더 잘하는 이유: direction loss가 dim을 명시적으로 강화?
- Channel-gate: channel attention이 direction dim을 선택적으로 증폭?
