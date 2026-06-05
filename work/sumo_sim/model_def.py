

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

CFG = dict(
    place             = "Nördlingen, Germany",
    n_rsu             = 20,
    n_dir             = 4,
    state_dim         = 8,
    seq_len           = 60,
    trend_horizon     = 60,
    action_dim        = 4,
    hidden            = 256,
    tcn_channels      = [128, 256],
    attn_heads        = 4,
    dropout           = 0.10,
    v2r_range         = 250,
    rsu_min_dist      = 300,
    gnb_range         = 800,

    # ── PSO ──
    meta_online_window   = 50,
    meta_update_freq     = 3,
    meta_online_iters    = 25,
    pso_particles        = 40,
    pso_w                = 0.65,
    pso_c1               = 1.8,
    pso_c2               = 1.8,

    # ── GA ──
    ga_pop_size       = 40,
    ga_crossover_rate = 0.80,
    ga_mutation_rate  = 0.20,
    ga_mutation_sigma = 0.10,
    ga_elite_frac     = 0.12,
)

DIR_NAMES  = ["North", "East", "South", "West"]
DIR_ARROWS = ["^", ">", "v", "<"]


# Model Architecture

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_c, out_c, k, dilation=1):
        super().__init__(in_c, out_c, k,
                         padding=(k - 1) * dilation, dilation=dilation)
    def forward(self, x):
        return super().forward(x)[..., :x.size(-1)]


class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_c, out_c, k, dilation), nn.GELU(),
            CausalConv1d(out_c, out_c, k, dilation), nn.GELU())
        self.res = (nn.Conv1d(in_c, out_c, 1)
                    if in_c != out_c else nn.Identity())
    def forward(self, x):
        return self.net(x) + self.res(x)


class RSUEncoder(nn.Module):
    def __init__(self, in_dim, tcn_channels, out_dim, dropout=0.1):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv1d(in_dim, tcn_channels[0], 3, padding=1), nn.GELU(),
            nn.Conv1d(tcn_channels[0], tcn_channels[0], 3, padding=1), nn.GELU())
        ch, layers = tcn_channels[0], []
        for i, oc in enumerate(tcn_channels):
            layers.append(TCNBlock(ch, oc, dilation=2**i)); ch = oc
        self.tcn  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(ch, out_dim), nn.GELU())
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.entry(x); x = self.tcn(x)
        return self.proj(self.pool(x).squeeze(-1))


class DirectionActorCritic(nn.Module):
    """Pure RL actor-critic. Actor = 4-class direction predictor."""
    def __init__(self, cfg):
        super().__init__()
        self.n_rsu  = cfg["n_rsu"]
        self.n_dir  = cfg["n_dir"]
        self.seq    = cfg["seq_len"]
        self.in_dim = cfg["state_dim"] + cfg["n_dir"]
        enc_out     = cfg["hidden"]
        drop        = cfg["dropout"]

        self.encoder = RSUEncoder(self.in_dim, cfg["tcn_channels"],
                                   enc_out, dropout=drop)
        self.attn = nn.MultiheadAttention(enc_out, cfg["attn_heads"],
                                           batch_first=True, dropout=drop)
        self.norm = nn.LayerNorm(enc_out)
        self.global_fc = nn.Sequential(
            nn.Linear(enc_out, enc_out), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(enc_out, enc_out // 2), nn.GELU())
        d = enc_out + enc_out // 2
        self.actor  = nn.Sequential(nn.Linear(d, d//2), nn.GELU(),
                                     nn.Dropout(drop),
                                     nn.Linear(d//2, cfg["action_dim"]))
        self.critic = nn.Sequential(nn.Linear(d, d//2), nn.GELU(),
                                     nn.Linear(d//2, 1))

    def _encode(self, flat):
        B, n, seq, d = flat.shape[0], self.n_rsu, self.seq, self.in_dim
        win = flat.view(B, seq, n, d).permute(0, 2, 1, 3)
        emb = self.encoder(win.reshape(B*n, seq, d)).view(B, n, -1)
        att, _ = self.attn(emb, emb, emb)
        emb    = self.norm(att + emb)
        agg    = emb.mean(dim=1)
        return torch.cat([agg, self.global_fc(agg)], dim=-1)

    def forward(self, x):
        z = self._encode(x)
        return self.actor(z), self.critic(z).squeeze(-1)

    def act(self, x):
        """Sample an action from the policy. Used during PPO rollout collection."""
        logits, val = self.forward(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), val

    def evaluate(self, x, a):
        """Evaluate actions under the current policy. Used during PPO update."""
        logits, val = self.forward(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(a), dist.entropy(), val

    def predict_direction(self, flat_state):
        """Convenience method for SUMO inference.
        Returns (dir_probs, pred_dir, raw_logits)."""
        with torch.no_grad():
            logits, _ = self.forward(flat_state)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            raw   = logits.squeeze(0).cpu().numpy()
            pred  = int(np.argmax(probs))
        return probs, pred, raw


# Online Metaheuristic Guide (PSO / GA)

class OnlineMetaheuristicGuide:

    def __init__(self, n_dir: int, method: str = "PSO", cfg: dict = None):
        cfg = cfg or {}
        self.n_dir       = n_dir
        self.method      = method.upper()
        self.window_size = cfg.get("meta_online_window", 30)
        self.update_freq = cfg.get("meta_update_freq",    5)
        self.n_iter      = cfg.get("meta_online_iters",  15)
        self.n_pop       = cfg.get("pso_particles",       30)

        self.pso_w   = cfg.get("pso_w",  0.65)
        self.pso_c1  = cfg.get("pso_c1", 1.8)
        self.pso_c2  = cfg.get("pso_c2", 1.8)

        self.ga_cx   = cfg.get("ga_crossover_rate", 0.80)
        self.ga_mut  = cfg.get("ga_mutation_rate",  0.20)
        self.ga_sig  = cfg.get("ga_mutation_sigma", 0.10)
        self.n_elite = max(1, int(self.n_pop * cfg.get("ga_elite_frac", 0.12)))

        self.bias = np.zeros(n_dir, dtype=np.float32)

        self._window: list = []
        self._step         = 0

        self._p_pos  = np.random.uniform(-0.5, 0.5, (self.n_pop, n_dir))
        self._p_vel  = np.zeros_like(self._p_pos)
        self._p_best = self._p_pos.copy()
        self._p_bsc  = np.zeros(self.n_pop)
        self._g_best = np.zeros(n_dir, dtype=np.float32)
        self._g_bsc  = 0.0

        self._ga_pop  = np.random.uniform(-0.5, 0.5, (self.n_pop, n_dir))
        self._ga_best = np.zeros(n_dir, dtype=np.float32)
        self._ga_bsc  = 0.0

        self.running_correct: list = []
        self.bias_snapshots:  list = []
        self.update_times_ms: list = []

    def _score(self, bias: np.ndarray) -> float:
        if not self._window:
            return 0.0
        correct = sum(
            int(np.argmax(logits + bias) == label)
            for logits, label in self._window
        )
        return correct / len(self._window)

    def predict(self, raw_logits: np.ndarray) -> int:
        return int(np.argmax(raw_logits + self.bias))

    def observe(self, raw_logits: np.ndarray, true_label: int):
        was_correct = int(np.argmax(raw_logits + self.bias) == int(true_label))
        self.running_correct.append(was_correct)

        self._window.append((raw_logits.copy(), int(true_label)))
        if len(self._window) > self.window_size:
            self._window.pop(0)

        self._step += 1

        if self._step % self.update_freq == 0 and len(self._window) >= 5:
            t0 = time.perf_counter()
            if self.method == "PSO":
                self._pso_update()
            else:
                self._ga_update()
            self.update_times_ms.append((time.perf_counter() - t0) * 1000)
            self.bias_snapshots.append(self.bias.copy())

    def _pso_update(self):
        n, d = self.n_pop, self.n_dir

        scores = np.array([self._score(p) for p in self._p_pos])

        improved          = scores > self._p_bsc
        self._p_best[improved] = self._p_pos[improved]
        self._p_bsc[improved]  = scores[improved]

        bi = int(np.argmax(self._p_bsc))
        if self._p_bsc[bi] > self._g_bsc:
            self._g_best = self._p_best[bi].copy()
            self._g_bsc  = self._p_bsc[bi]

        for _ in range(self.n_iter):
            r1 = np.random.rand(n, d)
            r2 = np.random.rand(n, d)
            self._p_vel = (
                self.pso_w * self._p_vel
                + self.pso_c1 * r1 * (self._p_best - self._p_pos)
                + self.pso_c2 * r2 * (self._g_best - self._p_pos)
            )
            self._p_pos = np.clip(self._p_pos + self._p_vel, -3.0, 3.0)

            scores = np.array([self._score(p) for p in self._p_pos])

            improved          = scores > self._p_bsc
            self._p_best[improved] = self._p_pos[improved]
            self._p_bsc[improved]  = scores[improved]

            bi = int(np.argmax(self._p_bsc))
            if self._p_bsc[bi] > self._g_bsc:
                self._g_best = self._p_best[bi].copy()
                self._g_bsc  = self._p_bsc[bi]

        self.bias = self._g_best.copy()

    def _ga_update(self):
        n, d = self.n_pop, self.n_dir

        for _ in range(self.n_iter):
            scores    = np.array([self._score(p) for p in self._ga_pop])
            elite_idx = np.argsort(scores)[-self.n_elite:]
            new_pop   = list(self._ga_pop[elite_idx])

            while len(new_pop) < n:
                k = min(3, n)
                t1 = np.random.choice(n, k, replace=False)
                t2 = np.random.choice(n, k, replace=False)
                c1 = int(t1[np.argmax(scores[t1])])
                c2 = int(t2[np.argmax(scores[t2])])

                if np.random.rand() < self.ga_cx:
                    alpha = np.random.rand(d)
                    child = alpha * self._ga_pop[c1] + (1.0 - alpha) * self._ga_pop[c2]
                else:
                    child = self._ga_pop[c1].copy()

                if np.random.rand() < self.ga_mut:
                    child = child + np.random.randn(d) * self.ga_sig

                new_pop.append(np.clip(child, -3.0, 3.0))

            self._ga_pop = np.array(new_pop[:n])

        final_scores  = np.array([self._score(p) for p in self._ga_pop])
        bi            = int(np.argmax(final_scores))
        self._ga_best = self._ga_pop[bi].copy()
        self._ga_bsc  = final_scores[bi]
        self.bias     = self._ga_best.copy()

    def rolling_accuracy(self, window: int = 100) -> np.ndarray:
        arr = np.array(self.running_correct, dtype=float)
        if len(arr) < window:
            return arr * 100
        return np.convolve(arr, np.ones(window) / window, mode="valid") * 100
