"""Frozen, shared year/month/day input composition; no invented dates."""
import numpy as np
import torch
from torch import nn
from .interval_context import IntervalEncoder
from .set_model import SetModel


class MaskedYMDTime(nn.Module):
    def __init__(self, components, dim, seed=732028):
        super().__init__()
        parts = np.asarray(components, dtype=np.int64)
        assert parts.ndim == 2 and parts.shape[1] == 4
        assert not parts[0].any()
        assert np.all((parts[:, 1] >= 0) & (parts[:, 1] <= 12))
        assert np.all((parts[:, 2] >= 0) & (parts[:, 2] <= 31))
        assert np.all(parts[:, 3] == (parts[:, :3] > 0).sum(1))
        self.embedding_dim = dim
        self.num_embeddings = len(parts)
        self.register_buffer('components', torch.from_numpy(parts))
        # Isolated initialization does not change Transformer/context RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.year = nn.Embedding(int(parts[:, 0].max()) + 1, dim, padding_idx=0)
            self.month = nn.Embedding(13, dim, padding_idx=0)
            self.day = nn.Embedding(32, dim, padding_idx=0)
            self.precision = nn.Embedding(4, dim, padding_idx=0)
        self.requires_grad_(False)

    def forward(self, ids):
        y, m, d, precision = self.components[ids].unbind(-1)
        value = self.year(y) + self.month(m) + self.day(d) + self.precision(precision)
        # Keep the per-coordinate initialization variance comparable across precisions.
        count = (precision + (precision > 0).long()).clamp_min(1)
        return value / count.to(value.dtype).sqrt()[..., None]


class BetaCalendarEncoder(IntervalEncoder):
    def __init__(self, protocol, events, train, dim=512, heads=8, time_mode='split'):
        assert time_mode == 'split'
        super().__init__(protocol, events, train, dim, heads, time_mode)
        self.times = MaskedYMDTime(protocol['ymd_components'], dim)


class BetaCalendarModel(SetModel):
    def __init__(self, protocol, events, train, dim=512, heads=8, time_mode='split'):
        nn.Module.__init__(self)
        self.n_source = protocol['entities'][0]
        self.encoder = BetaCalendarEncoder(protocol, events, train, dim, heads, time_mode)
