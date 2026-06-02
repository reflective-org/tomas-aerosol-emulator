"""Multi-head MLP for TOMAS aerosol emulation.

Shared backbone with separate output heads for different bin regions:
  - Nucleation head: bins 0-5   (Nk indices 0-5,   Mk indices 40-45)  -> 12 outputs
  - Core head:       bins 6-25  (Nk indices 6-25,  Mk indices 46-65)  -> 40 outputs
  - Coarse head:     bins 26-39 (Nk indices 26-39, Mk indices 66-79)  -> 28 outputs
  - Gc head:         index 80                                          -> 1 output

Each head is a small 2-layer MLP branching from the last backbone hidden layer.
Residual connection is applied per-head over the corresponding state indices.
"""

import torch
import torch.nn as nn


# Default bin region definitions (in terms of state vector indices).
# The boundary between core and coarse is configurable: pass `core_end_excl`
# to TomasMultiHeadMLP to shift it (e.g. core_end_excl=25 moves bin 25 from
# core to coarse).
NUCLEATION_IDX = list(range(0, 6)) + list(range(40, 46))    # 12 outputs
CORE_IDX = list(range(6, 26)) + list(range(46, 66))          # 40 outputs (default)
COARSE_IDX = list(range(26, 40)) + list(range(66, 80))       # 28 outputs (default)
GC_IDX = [80]                                                 # 1 output


def _build_partition(core_end_excl=26):
    """Return (nuc, core, coarse, gc) index lists for a given core/coarse split.

    core_end_excl: first bin that belongs to the coarse head (default 26).
      core_end_excl=25 moves bin 25 (and the corresponding Mk index 65) from
      the core head into the coarse head.
    """
    if not (7 <= core_end_excl <= 39):
        raise ValueError(f"core_end_excl must be in [7,39], got {core_end_excl}")
    nuc = list(range(0, 6)) + list(range(40, 46))
    core = list(range(6, core_end_excl)) + list(range(46, 40 + core_end_excl))
    coarse = list(range(core_end_excl, 40)) + list(range(40 + core_end_excl, 80))
    gc = [80]
    return nuc, core, coarse, gc


# Default reassembly order (matches default partition)
ALL_IDX = NUCLEATION_IDX + CORE_IDX + COARSE_IDX + GC_IDX
_IDX_TO_POS = {idx: pos for pos, idx in enumerate(ALL_IDX)}
REASSEMBLE_ORDER = [_IDX_TO_POS[i] for i in range(81)]


class HeadMLP(nn.Module):
    """Small 2-layer MLP head."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TomasMultiHeadMLP(nn.Module):
    """Shared backbone + 4 specialized output heads.

    Input: 90 (81 state + 9 params)
    Output: 81 (next state)

    Predicts residual per head: output_region = state_t_region + delta_region.
    """

    def __init__(self, hidden_dim=1024, n_backbone_layers=5,
                 head_hidden_dim=256, nuc_head_hidden_dim=None, dropout=0.0,
                 n_cond=9, core_end_excl=26):
        super().__init__()

        if nuc_head_hidden_dim is None:
            nuc_head_hidden_dim = head_hidden_dim

        self.n_cond = n_cond
        self.core_end_excl = core_end_excl

        nuc, core, coarse, gc = _build_partition(core_end_excl)
        reassemble = [(nuc + core + coarse + gc).index(i) for i in range(81)]

        # Shared backbone
        layers = []
        in_dim = 81 + n_cond
        for i in range(n_backbone_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Per-region heads (sizes depend on the chosen partition)
        self.head_nucleation = HeadMLP(hidden_dim, nuc_head_hidden_dim, len(nuc))
        self.head_core = HeadMLP(hidden_dim, head_hidden_dim, len(core))
        self.head_coarse = HeadMLP(hidden_dim, head_hidden_dim, len(coarse))
        self.head_gc = nn.Linear(hidden_dim, len(gc))  # single output, no hidden

        # Pre-register index tensors for extracting state regions
        self.register_buffer("nuc_idx", torch.tensor(nuc, dtype=torch.long))
        self.register_buffer("core_idx", torch.tensor(core, dtype=torch.long))
        self.register_buffer("coarse_idx", torch.tensor(coarse, dtype=torch.long))
        self.register_buffer("gc_idx", torch.tensor(gc, dtype=torch.long))
        self.register_buffer("reassemble", torch.tensor(reassemble, dtype=torch.long))

    def forward(self, x):
        """
        Args:
            x: (B, 90) concatenated [state_t_normalized, params_normalized]
        Returns:
            (B, 81) predicted next state (normalized)
        """
        state_t = x[:, :81]
        h = self.backbone(x)

        # Each head predicts a delta for its region
        delta_nuc = self.head_nucleation(h)
        delta_core = self.head_core(h)
        delta_coarse = self.head_coarse(h)
        delta_gc = self.head_gc(h)

        # Residual: add delta to corresponding state_t elements
        out_nuc = state_t[:, self.nuc_idx] + delta_nuc
        out_core = state_t[:, self.core_idx] + delta_core
        out_coarse = state_t[:, self.coarse_idx] + delta_coarse
        out_gc = state_t[:, self.gc_idx] + delta_gc

        # Reassemble into (B, 81) in original index order
        concat = torch.cat([out_nuc, out_core, out_coarse, out_gc], dim=-1)
        return concat[:, self.reassemble]
