# Background: Mathematical & Statistical Foundations

> FINAL_REPORT.md의 분석에 사용된 수학·선형대수·통계 배경지식 정리.
> 각 개념은 "정의 → 직관 → 본 분석에서의 역할" 순서로 서술.

---

## 목차

1. [Linear Algebra Foundations](#1-linear-algebra-foundations)
2. [Ridge Regression & R²](#2-ridge-regression--r)
3. [Logistic Regression & Cross-Entropy](#3-logistic-regression--cross-entropy)
4. [PCA (Principal Component Analysis)](#4-pca-principal-component-analysis)
5. [ANOVA F-statistic](#5-anova-f-statistic)
6. [Correlation & Per-dim Ranking](#6-correlation--per-dim-ranking)
7. [Cross-Validation](#7-cross-validation)
8. [Cosine Similarity & Frobenius Norm (Weight 비교)](#8-cosine-similarity--frobenius-norm)
9. [Standardization (Z-score)](#9-standardization-z-score)
10. [Null Ablation (Subspace Removal)](#10-null-ablation-subspace-removal)
11. [SigLip의 Position Embedding 수학](#11-siglip의-position-embedding-수학)
12. [LoRA Merge 수식](#12-lora-merge-수식)
13. [Set Overlap & Random Baseline](#13-set-overlap--random-baseline)

---

## 1. Linear Algebra Foundations

### 1.1 기호 약속

- 소문자 굵게: 벡터 (e.g. $\mathbf{x} \in \mathbb{R}^D$)
- 대문자 굵게: 행렬 (e.g. $\mathbf{X} \in \mathbb{R}^{N \times D}$, $N$ samples × $D$ dims)
- $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$: L2 norm
- $\|\mathbf{W}\|_F = \sqrt{\sum_{i,j} W_{ij}^2}$: Frobenius norm (행렬의 L2)
- $\mathbf{X}^\top$: transpose
- $\mathbf{I}_D$: $D \times D$ identity

### 1.2 Inner product & 선형 분리성

두 벡터의 inner product $\langle \mathbf{x}, \mathbf{w} \rangle = \mathbf{x}^\top \mathbf{w} = \sum_i x_i w_i$.

**Linear probe의 본질**: "어떤 정보가 linear로 decodable하다"는 것은, **어떤 weight vector $\mathbf{w}$가 존재해서 $\mathbf{x}^\top \mathbf{w}$가 target을 잘 예측**한다는 뜻. 즉 정보가 벡터 공간에서 **선형 경계로 분리 가능한 형태**로 존재.

### 1.3 Rank & Subspace

**Rank**: 행렬의 독립 열/행 개수. 정보가 $k$-dim subspace에 놓여있다 = rank $k$.

**본 분석과의 연결**: 
- "위치 정보가 low-rank subspace에 있나?" = "소수 주요 방향에 집중되어 있나?"
- Section B의 PCA 분석이 이를 직접 검증

---

## 2. Ridge Regression & R²

### 2.1 OLS → Ridge

**Ordinary Least Squares (OLS)**:
$$\mathbf{w}^* = \arg\min_\mathbf{w} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$$

Closed-form: $\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$.

문제: $\mathbf{X}^\top \mathbf{X}$가 singular / near-singular일 때 (고차원, 다중공선성) 불안정.

**Ridge regression** = OLS + L2 regularization:
$$\mathbf{w}^* = \arg\min_\mathbf{w} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$

Closed-form:
$$\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$

$\alpha$는 regularization 강도. $\alpha \to 0$이면 OLS, $\alpha \to \infty$이면 $\mathbf{w} \to 0$.

### 2.2 Bias term 처리

실제 모델은 $\hat{y} = \mathbf{x}^\top \mathbf{w} + b$. Bias column을 1로 붙여서 augmented $\mathbf{X}_b = [\mathbf{X}, \mathbf{1}]$로 푸되, **bias는 regularize하지 않음**:

```python
reg = α * I
reg[-1, -1] = 0  # bias 행은 0
w = solve(X_b.T @ X_b + reg, X_b.T @ y)
```

이유: bias는 "평균 shift"에 해당하므로 줄일 이유 없음.

### 2.3 R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

- $R^2 = 1$: 완벽 예측
- $R^2 = 0$: 평균으로 예측한 것과 같음 (chance baseline)
- $R^2 < 0$: 평균보다도 못함 (overfitting, generalization 실패)

**본 분석에서의 역할**: Position regression의 성능 지표. R² = 0.81 이면 "위치 분산의 81%를 linear projection으로 설명 가능" → 정보가 풍부하게 남아있음.

---

## 3. Logistic Regression & Cross-Entropy

### 3.1 Multinomial logistic regression

$C$-class 분류. 각 클래스에 대한 logit:
$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b} \in \mathbb{R}^C$$

Softmax로 probability 변환:
$$p_c = \frac{\exp(z_c)}{\sum_{c'} \exp(z_{c'})}$$

### 3.2 Cross-entropy loss

$$\mathcal{L} = -\sum_i \log p_{y_i} = -\sum_i z_{y_i} + \log \sum_{c} \exp(z_{i,c})$$

Gradient는 softmax output과 one-hot label의 차이: $\nabla_\mathbf{z} \mathcal{L} = \mathbf{p} - \mathbf{y}_\text{onehot}$.

### 3.3 본 분석에서의 쓰임

- Direction probe (4-way), Object probe (26-way), Letter probe (4-way) 모두 **linear + softmax + cross-entropy**
- 구현: `nn.Linear(D, C)`, Adam 300 epoch, full-batch
- 왜 linear만? → capacity 편향 제거. "정보가 linearly readable 한가?"를 묻는 실험

### 3.4 Chance level

$C$-class 균등이면 chance = $1/C$. 
- Direction 4-way: 0.25
- Object 26-way: 0.038
- Letter 4-way: 0.25 (shuffled이므로 독립)

---

## 4. PCA (Principal Component Analysis)

### 4.1 수학적 정의

데이터 $\mathbf{X} \in \mathbb{R}^{N \times D}$의 공분산 행렬 $\mathbf{\Sigma} = \frac{1}{N-1}\mathbf{X}_\text{centered}^\top \mathbf{X}_\text{centered}$.

**고유분해**: $\mathbf{\Sigma} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top$, where
- $\mathbf{V}$: 고유벡터 (principal components), 각 열이 한 방향
- $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots)$: 고유값 ($\lambda_1 \geq \lambda_2 \geq \ldots$)

**$k$-dim projection**: $\mathbf{X}_\text{PCA} = \mathbf{X} \mathbf{V}_{:,1:k}$.

### 4.2 Variance explained

$k$번째 PC가 설명하는 분산 비율:
$$\text{var ratio}_k = \frac{\lambda_k}{\sum_{j=1}^{D} \lambda_j}$$

Cumulative:
$$\text{cum var}_{1:k} = \frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{D} \lambda_j}$$

### 4.3 PCA의 편향

**PCA = 분산을 최대화하는 방향**. 즉 "데이터가 가장 퍼져있는 방향"을 찾음.

**중요한 함정**: "분산이 크다 = 중요하다"는 **틀린 가정**. 예를 들어 **class label을 구분하는 축은 분산이 작을 수도 있음**.

### 4.4 본 분석에서 드러난 현상 (Section B의 핵심)

obj_place에서:
- PCA top-50 (분산 92%)으로 position R²(x) = 0.03
- PCA top-100 (분산 96%, 단 4% 추가)에서 R²(x) = 0.70으로 점프

**해석**: 분산 92%를 차지하는 "dominant" 방향들은 **semantic content (object texture, background)** 를 인코딩. 위치 정보는 **분산은 작지만 의미있는 51~100번째 PC들**에 숨어있음.

이는 단순히 "PCA로 차원축소" 하는 전략이 위치 정보를 **완전히 파괴**할 수 있다는 경고.

---

## 5. ANOVA F-statistic

### 5.1 One-way ANOVA

여러 그룹 (클래스) 간 평균 차이를 한 dim에 대해 테스트.

- $C$개 그룹, 그룹 $c$의 샘플 수 $n_c$, 평균 $\bar{x}_c$
- 전체 평균 $\bar{x}$
- Between-group variance (SSB):
$$\text{SSB} = \sum_c n_c (\bar{x}_c - \bar{x})^2$$
- Within-group variance (SSW):
$$\text{SSW} = \sum_c \sum_{i \in c} (x_{i,c} - \bar{x}_c)^2$$

**F-statistic**:
$$F = \frac{\text{SSB}/(C-1)}{\text{SSW}/(N-C)}$$

### 5.2 해석

- $F$가 크면 → between-class 차이 >> within-class 차이 → **그 dim이 class를 잘 구분**
- $F \approx 1$이면 → class와 무관한 dim
- $F > 5, F > 10$ 등을 threshold로 "강한 신호 dim" 카운트

### 5.3 본 분석에서의 역할

- **각 dim에 대해 F-stat 계산** (one-way ANOVA over 4 directions on delta vector)
- Top-50 direction-discriminative dim 선택: `argsort(-F)[:50]`
- 모델 비교: vanilla vs baseline vs ours의 top-50 F-stat 평균
- **F-stat이 작은 변화로도 크게 움직이는 이유** (Section G):
  - Weight 5% 변화, row cosine > 0.98 → 방향은 같음
  - But class별 centroid 미세 이동 → between-class variance 증가
  - → F-stat 상승 (selectively sharper class separation)

---

## 6. Correlation & Per-dim Ranking

### 6.1 Pearson correlation

두 변수 사이 선형 관계:
$$\text{corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2 \cdot \sum_i (y_i - \bar{y})^2}}$$

범위 $[-1, 1]$. $|corr|$가 클수록 선형 예측 가능.

### 6.2 Multi-target correlation (Section B의 position ranking)

Position은 2D $(x, y)$이므로 **joint magnitude**로 순위:
$$\text{corr}_\text{pos}[d] = \sqrt{\text{corr}(X_{:,d}, \text{pos}_x)^2 + \text{corr}(X_{:,d}, \text{pos}_y)^2}$$

직관: 두 축 중 어느 하나와라도 강하게 상관되면 "position-encoding dim"으로 간주.

$$\text{top}_k = \text{argsort}(-\text{corr}_\text{pos})[:k]$$

---

## 7. Cross-Validation

### 7.1 K-fold CV

데이터를 $K$개 fold로 나누고, 매번 1개를 test, 나머지 $K-1$개를 train. $K$번 반복해서 평균.

**KFold** (regression): 무작위 분할
**StratifiedKFold** (classification): 각 fold에서 **class 비율 보존**

### 7.2 왜 fold마다 scaler 재학습?

```python
for train_idx, test_idx in kfold.split(X):
    scaler = StandardScaler().fit(X[train_idx])  # train에서만 fit
    X_tr = scaler.transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])         # test는 그냥 transform
```

**이유**: Test set의 분포 정보가 train에 leak되면 metric이 낙관적으로 편향됨 (data leakage).

### 7.3 본 분석에서의 설정

- 5-fold
- α = 1.0 for Ridge
- 300 epoch Adam for Logistic probe

---

## 8. Cosine Similarity & Frobenius Norm

### 8.1 Cosine similarity

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} \in [-1, 1]$$

**"방향이 얼마나 같은가"**를 측정 (norm 무시).
- $\cos = 1$: 완전 동일 방향
- $\cos = 0$: 직교
- $\cos = -1$: 반대 방향

### 8.2 Frobenius norm & relative difference

$\|\mathbf{W}\|_F = \sqrt{\text{tr}(\mathbf{W}^\top \mathbf{W})} = \sqrt{\sum_{i,j} W_{ij}^2}$

**Relative difference**:
$$\text{rel\_diff}(\mathbf{W}_t, \mathbf{W}_v) = \frac{\|\mathbf{W}_t - \mathbf{W}_v\|_F}{\|\mathbf{W}_v\|_F}$$

얼마나 많이 변했는가를 원본 크기에 정규화.

### 8.3 Row-wise cosine (Section G의 핵심)

Projector weight $\mathbf{W} \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$. 각 행 $\mathbf{w}_i$는 "output dim $i$를 만드는 linear combination 방향".

```python
row_cos[i] = cos(W_v[i, :], W_t[i, :])
row_cos_min = min(row_cos)
```

**의미**:
- `rel_diff` = 4.7% **AND** `row_cos_min` > 0.98
- → Weight가 전체적으로 5% 커졌지만 **각 output dim을 만드는 방향은 거의 안 바뀜**
- → 변한 건 주로 **크기 (norm)**, 방향 아님

**본 분석 결론**: Ours method는 **projector의 "의미 구조"를 보존하면서 scale만 미세 조정** → semantic 손상 없이 direction 신호 증폭.

---

## 9. Standardization (Z-score)

$$x'_{i,d} = \frac{x_{i,d} - \mu_d}{\sigma_d}$$

각 dim을 평균 0, 표준편차 1로 정규화.

**왜 필요?**:
- Ridge regression의 $\alpha$가 모든 dim에 동일하게 적용되는데, dim마다 scale이 다르면 특정 dim만 과도하게 regularize됨
- PCA는 분산 기준인데, dim scale 차이가 주성분 방향을 왜곡
- 본 분석: `StandardScaler`를 fold마다 fit

---

## 10. Null Ablation (Subspace Removal)

### 10.1 아이디어

"Dimension $S \subset \{0, \ldots, D-1\}$를 제거하면 정보가 얼마나 남아있나?"

단순 dim drop:
$$\mathbf{X}_\text{ablated} = \mathbf{X}_{:, \bar{S}}, \quad \bar{S} = \{0, \ldots, D-1\} \setminus S$$

### 10.2 더 정교한 방법 (본 분석에선 단순 drop 사용)

Projection ablation: 문제의 subspace를 **orthogonal projection으로 제거**
$$\mathbf{X}_\text{ablated} = \mathbf{X}(\mathbf{I} - \mathbf{V}_S\mathbf{V}_S^\top)$$

### 10.3 본 분석에서의 쓰임

- Top-$k$ position-correlated dim을 제거하고, 남은 $(D-k)$ dim으로 probe
- Top-500 제거 후 R²(x) = 0.41 남음 → **정보가 distributed**

**중요**: ablation은 "이 dim이 **유일한** 정보원인가"를 묻는 게 아니라 "**강한** 정보원이 제거된 상태에서 나머지가 얼마나 보완하는가"를 측정.

---

## 11. SigLip의 Position Embedding 수학

### 11.1 Patch embedding

Input image $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$을 $P \times P$ patch로 분할 (여기선 $P=14$, $H=W=384$ → 27×27 = 729 patches).

각 patch $\mathbf{p}_i \in \mathbb{R}^{3 \times P \times P}$를 flatten 후 linear projection:
$$\mathbf{e}_i = \mathbf{W}_\text{patch} \cdot \text{flatten}(\mathbf{p}_i) + \mathbf{b}_\text{patch} \in \mathbb{R}^{D}$$

### 11.2 Absolute position embedding

학습된 **position table** $\mathbf{E}_\text{pos} \in \mathbb{R}^{729 \times D}$.

$$\mathbf{z}_i^{(0)} = \mathbf{e}_i + \mathbf{E}_\text{pos}[i]$$

**핵심 함의**: 같은 patch content라도 **다른 index $i$에 오면 다른 $\mathbf{z}_i^{(0)}$**.

→ 이 시점에서 이미 **translation equivariance가 깨짐**.

### 11.3 Self-attention propagation

Transformer layer에서:
$$\mathbf{z}_i^{(\ell+1)} = \mathbf{z}_i^{(\ell)} + \sum_j \alpha_{ij} \mathbf{V}\mathbf{z}_j^{(\ell)}$$

Attention weight $\alpha_{ij}$가 모든 patch를 섞음 → 위치 정보가 **729개 전 patch로 전역 전파**.

### 11.4 Mean-pool 후 위치 잔존 (수학적으로 왜?)

$$\bar{\mathbf{z}} = \frac{1}{729} \sum_i \mathbf{z}_i^{(L)}$$

각 $\mathbf{z}_i^{(L)}$가 position-dependent한 함수의 결과이므로, 평균도 content/layout에 대한 **통계적 요약**.

**구체적으로**: 물체가 좌상단에 있을 때와 우하단에 있을 때:
- 각 patch의 `position_emb` 값이 다름
- Self-attention으로 "어느 patch가 object의 중심인지" 정보가 전파
- Mean-pool 후에도 이 bias가 남음 → linear probe로 복원 가능

---

## 12. LoRA Merge 수식

### 12.1 LoRA (Low-Rank Adaptation)

원본 weight $\mathbf{W}_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$에 low-rank update:
$$\mathbf{W} = \mathbf{W}_0 + \Delta\mathbf{W}, \quad \Delta\mathbf{W} = \frac{\alpha}{r} \mathbf{B}\mathbf{A}$$

where
- $\mathbf{A} \in \mathbb{R}^{r \times d_\text{in}}$ (random init)
- $\mathbf{B} \in \mathbb{R}^{d_\text{out} \times r}$ (zero init)
- $r \ll \min(d_\text{in}, d_\text{out})$: rank (본 분석: $r=64$)
- $\alpha$: scaling factor (본 분석: $\alpha=128$)
- Effective scale: $\alpha/r = 2.0$

### 12.2 왜 low-rank?

- Full fine-tuning 파라미터 수: $d_\text{out} \cdot d_\text{in}$
- LoRA: $r(d_\text{out} + d_\text{in})$
- $d = 3584$, $r = 64$이면 56배 감소

### 12.3 Merge 수식 (inference 시)

학습 후엔 $\Delta\mathbf{W}$를 precompute해서 base에 합침:
$$\mathbf{W}_\text{merged} = \mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$$

본 분석 (Section H): 196개 attention layer에 대해 수동 merge:
```python
delta = (alpha / r) * (B @ A)  # (out, in)
W_merged = W_base + delta
```

---

## 13. Set Overlap & Random Baseline

### 13.1 Top-k overlap

두 top-k 집합 $S_1, S_2 \subset \{0, \ldots, D-1\}$의 교집합 크기:
$$\text{overlap} = |S_1 \cap S_2|$$

### 13.2 Random baseline (hypergeometric 기대값)

두 집합이 독립적 무작위 선택이면:
$$\mathbb{E}[|S_1 \cap S_2|] = \frac{k \cdot k}{D} = \frac{k^2}{D}$$

본 분석:
- $k=50$, $D=1152$ (pre-proj): 기대값 ≈ 2.2
- $k=50$, $D=3584$ (post-proj): 기대값 ≈ 0.7

### 13.3 본 분석에서의 해석

- Position ∩ Direction = 17/50 → 기대값 2.2 대비 **8배 과잉** → **강한 공유**
- Position ∩ Object = 1/50 → 기대값 정도 → **공유 없음** (분리된 subspace)

---

## Appendix: 주요 식 요약표

| 개념 | 식 |
|---|---|
| Ridge | $(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| R² | $1 - \text{SS}_\text{res} / \text{SS}_\text{tot}$ |
| Cross-entropy | $-\sum_i \log p_{y_i}$ |
| Softmax | $p_c = e^{z_c} / \sum_{c'} e^{z_{c'}}$ |
| PCA | $\mathbf{\Sigma} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top$ |
| F-stat | $\frac{\text{SSB}/(C-1)}{\text{SSW}/(N-C)}$ |
| Pearson corr | $\text{Cov}(X,Y) / (\sigma_X \sigma_Y)$ |
| Cosine | $\mathbf{u}^\top\mathbf{v} / (\|\mathbf{u}\|\|\mathbf{v}\|)$ |
| Frobenius | $\sqrt{\sum_{ij} W_{ij}^2}$ |
| Rel diff | $\|\mathbf{W}_t - \mathbf{W}_v\|_F / \|\mathbf{W}_v\|_F$ |
| LoRA merge | $\mathbf{W}_0 + (\alpha/r)\mathbf{B}\mathbf{A}$ |
| Overlap E | $k^2/D$ |
