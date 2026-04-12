# Implementation Details

## Shared Components

### Features Used
- **Feature file**: `features/R2R_real_color.npz`
  - `pre_mean`: (8000, 8, 1152) — SigLip 마지막 layer의 pre-projection tokens, N=729 패치 평균
  - `post_mean`: (8000, 8, 3584) — Projector 통과 후, N=729 패치 평균
- **Metadata**: 8000 videos, 20 objects × 4 directions × 100 each
  - 20 objects: car, chair, dining table, person, cup, bowl, bottle, book, truck, motorcycle, umbrella, broccoli, banana, bench, sheep, potted plant, handbag, cake, cow, boat
  - 4 directions: up, down, left, right (각 2000개)
  - 8 frames per video, 384×384, solid color bg, linear motion

### SigLip & Projector
- **Vision encoder**: `google/siglip-so400m-patch14-384` (마지막 layer 1개 제거, head=Identity)
- **Projector** (mlp2x_gelu): `Linear(1152→3584) → GELU → Linear(3584→3584)`
- **Mean pooling**: N=729 패치에 대해 `.mean(dim=1)` → (T, D)

### GPU Probe 구현
- **Device**: `cuda:0` (CUDA_VISIBLE_DEVICES=5로 GPU 5 사용)
- **Ridge regression**: 닫힌 형태 `w = (X'X + αI)⁻¹ X'y`, α=1.0, bias regularize 안 함
- **Classification**: `nn.Linear` + Adam (lr=0.01, weight_decay=1e-4), 300 epochs, full-batch
- **Cross-validation**: 5-fold, shuffle=True, random_state=42
- **Scaling**: `StandardScaler` (fit on train, transform on test) fold별로

---

## Phase 1: Position Subspace Analysis

### Exp 1.1: PCA Dimensionality
- **Input**: pre_mean (8000, 8, 1152) → reshape to (64000, 1152)
- **Subsample**: random 10000 out of 64000 (seed=42) — full PCA가 메모리 부담
- **PCA**: `sklearn.decomposition.PCA(n_components=1000)` (1152까지 가능하지만 1000으로 제한)
- **Target**: frame의 실제 (x, y) 픽셀 좌표 (metadata의 positions, (8, 2))
- **Probe**: 각 k ∈ {1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000}에 대해 첫 k개 PC만 써서 Ridge regression
- **Metric**: 5-fold CV R² on (x), (y) 별도

### Exp 1.2: Top-k Dim Selection
- **Position dims**: per-dim `|corrcoef(X[:,d], pos_x)|²  + |corrcoef(X[:,d], pos_y)|²`의 제곱근으로 score 계산, 상위 k개
- **Object dims**: per-dim one-way ANOVA F-statistic across 20 object classes (mid frame, N=8000), 상위 k개
- **Random dims**: `np.random.permutation(1152)[:k]` (seed=42)
- **Probe**: 각 dim set으로 Ridge regression → position R2 비교
- **k values tested**: {10, 20, 50, 100, 200}

### Exp 1.3: Position-Nulled Ablation
- **Top position dims** 선정: Exp 1.2와 동일 (|correlation| score 상위 k개)
- **"Nulled" features**: top-k dim을 제외한 나머지 (1152-k) dim 사용
  ```python
  remaining_dims = [d for d in range(1152) if d not in top_pos]
  X_nulled = X[:, remaining_dims]
  ```
- **3가지 probe**:
  - Object class (20-way): mid frame (t=4), 8000 samples, 300 epochs LR
  - Direction (4-way): delta = last − first frame, 8000 samples, 300 epochs LR
  - Position R2(x): frame-level, 10000 subsample, Ridge closed-form
- **k values**: {10, 20, 50, 100, 200} + full (k=0 reference)

---

## Phase 2: Complexity Gradient

### Data
- Testbed (200 videos each, 32 frames → 8 sampled): shape_color, shape_place, obj_color, obj_place
- Frame sampling: `np.linspace(0, 31, 8, dtype=int)` = [0, 4, 8, 13, 17, 22, 26, 31]
- Position computation: metadata의 `start_pos`, `direction`, `speed`로 `pos_t = start + direction_vec × speed × t` (선형)
- R2R_real_color: 8000 videos, 8 frames (real metadata positions 사용)

### Exp 2.1: Probes per Condition
- **Position R2**: frame-level (1600 samples for 200-vid subsets; 10000 subsample for 8000-vid)
- **Delta direction**: `feat[:, -1] - feat[:, 0]`, video-level
- **Object class**: mid frame (t=3 or t=4), video-level
- **F_dir**: one-way ANOVA F-stat per dim on delta (4 direction groups)
  - `top50_mean`: 상위 50 dim의 F-stat 평균

### Exp 2.3: F-stat Distribution
- Testbed 200-vid subsets와 R2R_real_color 8000-vid 간 직접 비교는 **sample size 영향으로 불가능** (F-stat ∝ N)
- Testbed 4개만 서로 비교: thresholds {1, 5, 10, 50, 100}에서 F-stat 넘는 dim 수 카운트

---

## Phase 3: LLM Hidden State Probing

### Model Loading
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
model = LlavaQwenForCausalLM.from_pretrained(
    MODEL_BASE, low_cpu_mem_usage=True,
    torch_dtype=torch.float16, device_map="cuda:0",
)
```
- fp16, ~14GB VRAM

### Video Selection
- 400 videos from R2R_real_color: 100 per direction (balanced)
- `np.random.seed(42)` → `np.random.choice(len(dir_vids), 100, replace=False)` per direction

### Prompt (중요 버그 수정)
- Template: `qwen_1_5` conversation
- Prompt: `<image>\nDescribe the video.`
- **반드시 `tokenizer_image_token` 사용** (단순 `tokenizer()` 쓰면 `<image>`가 일반 text로 토크나이즈되어 vision token 병합 안 됨!)
  ```python
  from llava.mm_utils import tokenizer_image_token
  from llava.constants import IMAGE_TOKEN_INDEX
  input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
  ```

### Hidden State Extraction
```python
with torch.no_grad():
    # Merge vision tokens into input embeddings
    (_, _, attn, _, inputs_embeds, _) = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None,
        images=[video_tensor], modalities=["video"]
    )
    outputs = model.forward(
        input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attn,
        output_hidden_states=True, return_dict=True
    )
# outputs.hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, 3584)
# hidden_states[0] = embedding output (no attention applied)
# hidden_states[1] = after layer 0 (one transformer block)
# ...
# hidden_states[28] = after layer 27 (last)
last_token_hidden = hidden_states[layer_idx][0, -1, :].detach().cpu().float().numpy()
```

### Layers Probed
- `[0, 1, 4, 8, 12, 16, 20, 24, 28]` (Qwen2-7B has 28 decoder layers)

### Probes on Last Token Hidden (D=3584)
- **Direction**: 4-way classification, LR 300 epochs, 5-fold CV
- **Object**: 20-way classification, LR 300 epochs, 5-fold CV
- **Position R2(x)**: Ridge regression on frame-4 position (mid frame 기준)
- All on 400 samples (video-level; 각 비디오당 last token 1개)

---

## Phase 4-5: Trained Model Comparison

### Models
- **Vanilla**: `model.safetensors` in LLaVA-Video-7B-Qwen2 (base)
- **Baseline / delta_direct / channel_gate**: `4combo_new/work_dirs/*/non_lora_trainables.bin`
  - mm_tunable_parts: `mm_mlp_adapter, mm_language_model` (projector 학습됨!)

### Projector Loading
```python
# Trained projector from non_lora_trainables.bin
state = torch.load(non_lora_path, map_location="cpu")
proj_state = {k.split("mm_projector.")[-1]: v for k, v in state.items() if "mm_projector" in k}
projector.load_state_dict(proj_state)
```

### ⚠️ Critical Approximation
- 저장된 `pre_mean`은 mean-pooled (8000, 8, 1152)
- 진짜 post-proj: `projector(full patches)`를 mean-pool (729개 패치마다 개별 MLP 통과 후 평균)
- 우리가 계산한 것: mean-pool된 것에 projector 적용 (`projector(mean(patches))`)
- **GELU 비선형성 때문에 둘이 다름**: `mean(GELU(Wx)) ≠ GELU(mean(Wx))`
- 정확한 비교는 full (8, 729, 1152) features 재추출 후 각 patch에 projector 적용해야
- 하지만 **모델 간 상대 비교 (vanilla vs baseline vs delta_direct)는 bias가 consistent하므로 유효**

### Probes
- Position R2: frame-level, 10000 subsample
- Delta direction: video-level, 8000 samples
- Object class: mid frame, video-level
- **Direction F-stat**: per-dim ANOVA on delta vectors
- **Top-50 overlap with vanilla**: `len(set(top50_model) & set(top50_vanilla))`
- **F increase/decrease**: `f_stats_model - f_stats_vanilla` 각 dim별로, 상위 20개 출력

---

## Phase 5b: Real-world (SSv2, KTH)

### Data
- **SSv2**: `/local_datasets/vlm_direction/ssv2_VP/ssv2_VP_default/{left, right}` — 각 250 videos
- **KTH**: `/local_datasets/vlm_direction/KTH_VP_lr/KTH_VP_default/{left, right}` — left 447, right 452
- 2-way classification (left/right), chance = 50%

### Feature Extraction
- Load all frames, uniform sample 8 frames if longer
- SigLip + Projector (vanilla) on-the-fly
- Mean-pool over patches → (T, 1152) pre, (T, 3584) post

### Probes
- Delta direction, single frame (mid), T-mean — all 2-way, full sample count
- Per-dim F-stat on delta, count dims with F>5, F>10

---

## Cross-Validation Implementation Detail

모든 CV probe에서:
1. **Scaler는 fold마다 새로 fit** (train에 fit, test에 transform)
2. **Random state 고정**: train/test split reproducible
3. **StratifiedKFold for classification** (class balance 유지), KFold for regression
4. **평균 ± 표준편차** 보고 (보통 std < 0.01로 매우 안정적)

---

## GPU Resource Usage

| Phase | GPU | Peak VRAM | Runtime |
|---|---|---|---|
| Phase 1 | 1 (GPU 5) | ~2GB | ~15 min |
| Phase 2 | 1 (GPU 5) | ~2GB | ~10 min |
| Phase 3 | 1 (GPU 0) | ~16GB (7B fp16 + activations) | ~15 min |
| Phase 4 | 1 (GPU 0) | ~5GB (projector + features) | ~3 min |
| Phase 5b | 1 (GPU 0) | ~4GB (SigLip + projector) | ~10 min |

모든 probe는 nn.Linear + Adam, 300 epochs, full-batch (8000×1152 fits easily in 20GB budget)
