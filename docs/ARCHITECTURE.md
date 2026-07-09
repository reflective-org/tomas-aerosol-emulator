# Axial Trajectory Operator — Architecture, Explained Piece by Piece

Model: `AxialTrajOperator` in `model_axial_traj.py`.
Released config (`checkpoints/axial_aug5`): `d_model=512`, `nhead=8`,
`num_blocks=8`, `dim_ff=768`, `ff_experts=2`, ~29.5M params.

---

## 0. What the model is, at a high level

A **non-autoregressive trajectory operator**: given one air parcel's aerosol state at t=0 plus
the 6-hour forcing, it predicts the entire 6-hour trajectory — 12 states at 30, 60, …, 360 min —
in a **single forward pass**. It never feeds its own predictions back in.

The central object it operates on is a **2-D grid of tokens: (12 time-steps) × (41 tokens)**, and
the core idea is **axial attention** over that grid.

---

## 1. State representation & why log-space

The aerosol state is **81 numbers**: 40 `Nk` (number concentration per size bin) + 40 `Mk`
(sulfate mass per size bin) + 1 `Gc` (H2SO4 gas). The "two moments" (number + mass per bin) are
exactly what TOMAS tracks — TOMAS = TwO-Moment Aerosol Sectional.

These span ~20 orders of magnitude, so the model works in **log-space**: `log(max(state, 1e-30))`,
then **z-scored** per-component (subtract mean, divide std) via `NormStatsV4`. Every input to the
network is therefore a normalized log value ~O(1). The network predicts in this normalized-log
space; we only exponentiate back to physical units at the very end.

---

## 2. Tokenization — 81 numbers -> 41 tokens (forward, lines 78-82)

```python
Nk0, Mk0, Gc0 = state[:40], state[40:80], state[80:81]
bin_feat = stack([Nk0, Mk0], -1)          # (B, 40, 2)  — each size bin = (number, mass)
gas_feat = cat([Gc0, 0])                  # (B, 1, 2)   — gas token, second slot zero-padded
tok0     = cat([bin_feat, gas_feat], 1)   # (B, 41, 2)
x        = tok_in(tok0)                    # (B, 41, 512)  Linear(2 -> 512)
```

- **41 tokens** = 40 size bins + 1 gas token. Each *bin* token carries a 2-vector `(Nk, Mk)` — its
  number and mass together, so the model sees them as one coupled entity. The gas is a 41st token
  `(Gc, 0)`.
- `tok_in` is a single `Linear(2 -> 512)`: lifts each token's 2 physical features into a 512-dim
  representation the attention layers work with.

Key framing: **a size bin is a "token," like a word in a sentence.** Attention then learns which
bins talk to which.

---

## 3. Laying out the time grid & injecting forcing (lines 83-85)

```python
x = x.unsqueeze(1).expand(B, 12, 41, 512)             # copy initial state to all 12 horizons
x = x + bin_emb[None,None] + time_emb[None,:,None]    # learned position embeddings
x = x + cond_in(cond[:, 1:13]).unsqueeze(2)           # forcing per output time
```

1. **Broadcast across time.** The same embedded initial state is copied into all 12 output
   time-slots. The network transforms slot k into "the state at horizon k." (This is what makes it
   non-autoregressive — all 12 horizons are built in parallel from t=0.)

2. **Positional embeddings.** Attention is permutation-invariant, so without position info bin 5 vs
   bin 30 (and 30 min vs 6 h) would be indistinguishable. Two learned tables fix that:
   - `bin_emb` — shape `(41, 512)`: a learned vector per token (which bin / the gas).
   - `time_emb` — shape `(12, 512)`: a learned vector per horizon.

3. **Forcing injection.** `cond_in` is `Linear(6 -> 512)`. The 6 conditions — **Pressure, NH3,
   Temperature, RH, H2SO4 vapor, fion** — at each of the 12 output times are projected to 512-dim
   and **added to every token at that time-slot** (`unsqueeze(2)` broadcasts the same forcing vector
   across all 41 tokens of that step). So each grid cell = "initial state + which bin + which time +
   the environment at that time." (`fion`, the ion-pair formation rate, is the 6th input; its
   `cond_in` column was zero-initialized on warm-start so the augmentation-tuned model started
   identical to its 5-channel base and learned `fion` from there.)

After this, `x` is the grid `(B, 12, 41, 512)` that the stack of blocks refines.

---

## 4. The AxialBlock — the heart of the model (lines 41-57)

Each of the 8 blocks does **three sub-layers**, each wrapped in pre-LayerNorm + a residual
connection (`x = x + sublayer(LN(x))`, the standard stable-transformer pattern):

### (a) Bin-axis attention (lines 52-53)
```python
h = ln1(x).reshape(B*T, P, d)       # treat each (batch, time) as a sequence of 41 tokens
a, _ = bin_attn(h, h, h)            # self-attention over the 41 tokens
x = x + a.reshape(B, T, P, d)
```
For each fixed time-step, the 41 tokens attend to each other — all-pairs over bins+gas. This is
where size bins couple: bin i can pull information from bin j. **Physics rationale:** coagulation is
a pairwise Ni*Nj process (any two bins can collide and feed a third) — exactly what all-pairs
bin-axis attention expresses. It's also where the gas token influences the bins
(condensation / nucleation source).

### (b) Time-axis attention (lines 54-55)
```python
h = ln2(x).permute(0,2,1,3).reshape(B*P, T, d)   # treat each (batch, token) as a sequence of 12 steps
a, _ = time_attn(h, h, h)                          # bidirectional self-attention over time
x = x + a.reshape(...).permute(...)
```
For each fixed bin, its 12 time-slots attend to each other. It's **bidirectional** (non-causal):
6 h can look at 1 h and vice-versa. This is the non-autoregressive temporal composition — the model
reasons about the whole trajectory at once with the full forcing in view, instead of stepping one Δt
at a time. The permute/reshape is just plumbing so PyTorch's `MultiheadAttention` attends over the
right (sequence) axis: first bins are the sequence, then time is the sequence.

### (c) Feed-forward (line 56)
```python
x = x + ff(ln3(x))
```
A position-wise MLP applied to every token independently. With `ff_experts=2` this is the MoE below;
otherwise a plain `Linear(512->768) -> GELU -> Linear(768->512)`.

**Why "axial" instead of full 2-D attention?** Full attention over all 12x41 = 492 cells would be
492^2 ~ 242k pairs per layer. Axial factorizes it into 41^2 (bins) + 12^2 (time) — far cheaper, and a
meaningful inductive bias: the two axes are physically distinct (which-particle-size vs which-time),
so giving each its own attention is both efficient and natural. (A prior time-only attention model
failed for lack of bin coupling; bin-axis attention is the fix.)

### Inside each attention (`nn.MultiheadAttention`, 8 heads)
Standard multi-head scaled-dot-product:
- Project the input to **Queries, Keys, Values**, split into **8 heads** (512/8 = 64 dims each).
- Per head: `softmax(Q*K^T / sqrt(64)) * V` — each token forms a softmax-weighted average of the
  others, weighted by dot-product similarity.
- Concatenate the 8 heads -> linear output projection back to 512.

8 heads learn 8 relationships in parallel (e.g. one head tracks neighbor-bin growth, another the
gas -> nucleation-bin link).

---

## 5. The Mixture-of-Experts FFN (lines 28-38)

```python
g    = softmax(gate(x))                          # (..., 2) routing weights per token
outs = stack([expert(x) for expert in experts])  # 2 independent MLPs
return (g * outs).sum                            # weighted blend
```
Two expert MLPs and a learned **gate** producing a softmax weight per token; the output is a weighted
blend of the two experts (a *soft* MoE — both run, then blend, not hard top-k). **Rationale:** the
gate can switch sharply between regimes — e.g. a "calm decay" expert vs a "nucleation burst" expert —
a threshold-friendly inductive bias for the very nonlinear burst onset.

---

## 6. Residual head & decoding (lines 70-71, 88-90)

```python
delta = head(norm(x))                       # Linear(512 -> 2), per token
return state0 + delta                        # persistence-residual
```
- `head` maps each token's 512-dim vector to a **2-vector = (dNk, dMk)**; the gas token's first slot
  gives dGc.
- **The head is zero-initialized** (weights AND bias). So at the start of training `delta = 0` and the
  output is exactly `state0` — the model **begins as the identity / "persistence" map** ("nothing
  changes over 6 h"). That's a deliberately good basin: from-scratch models tend to collapse to a
  biased degenerate solution (the "Jensen trap"); starting at persistence and learning only the
  correction Δ avoids that.
- Output per horizon: `state(t) = state(0) + Δ(t)`, in normalized-log space. Final step (outside the
  model): denormalize -> log-state -> `exp` -> physical `Nk, Mk, Gc`.

So the network's actual job is to predict **how much each bin's log-number and log-mass change** from
t=0 to each of the 12 horizons, given the forcing.

---

## 7. The non-autoregressive property (and why it matters)

A single `forward` produces all 12 states. There is **no rollout** — the model never consumes its own
output. Historically the dominant error in autoregressive versions was *rollout amplification* (small
per-step biases compounding over hundreds of steps). Predicting the whole trajectory at once with
bidirectional time-attention removes that, which is why the error is roughly **flat across horizons**
rather than growing — the 3 h prediction is about as good as the 30 min one.

---

## 8. How training shapes it (brief)

- **Loss** is computed in denormalized log-space: a Huber loss on (predicted - true) log-state,
  **bin-weighted** (nucleation/gas bins up-weighted), plus a small auxiliary MSE in normalized space
  and a clipped **physical relative-error** term so the metric we care about (per-bin rel-err) is in
  the objective. See `train_axial_aug.py`.
- **Warm-start + persistence-residual** mean it starts at a sensible map and refines.
- **Normalization is frozen from the base model** during the augmentation fine-tune, which is what
  kept augmentation from corrupting in-distribution performance. See `scripts/train_axial_aug5.sh`.

---

## Putting it together (one line per stage)

initial 81-dim log-state
  -> **41 tokens** (40 bins x (N,M) + gas)
  -> embed to 512 and **copy across 12 horizons** + add bin/time position + **inject forcing per step**
  -> **8x [ bin-attention -> time-attention -> MoE-FFN ]** refine the (12 x 41) grid
  -> **zero-init residual head** predicts dlogN, dlogM per cell
  -> `state(t) = state(0) + Δ`
  -> exp
  -> physical 6-hour trajectory.

---

## Tensor-shape cheat sheet

| stage                         | shape                  |
|-------------------------------|------------------------|
| input state (log, normalized) | (B, 81)                |
| tokens                        | (B, 41, 2)             |
| after tok_in embed            | (B, 41, 512)           |
| broadcast over time           | (B, 12, 41, 512)       |
| + forcing cond_in(6->512)     | (B, 12, 41, 512)       |
| each AxialBlock (x8)          | (B, 12, 41, 512)       |
| head delta                    | (B, 12, 41, 2)         |
| output trajectory             | (B, 12, 81)            |

Glossary: B = batch; 12 = output time-steps (Δt=30 min to 6 h); 41 = tokens (40 size bins + 1 gas);
2 = (number, mass) per token; 512 = model width; 6 = forcing channels (P, NH3, T, RH, H2SO4, fion);
8 = attention heads / transformer blocks.
