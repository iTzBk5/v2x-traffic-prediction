import os, sys, math, random, warnings, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import Counter
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym
warnings.filterwarnings("ignore")

SUMO_SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sumo_sim")
if os.path.isdir(SUMO_SIM_DIR):
    sys.path.insert(0, os.path.abspath(SUMO_SIM_DIR))
import v2x_gym
from v2x_gym.envs.direction_pred_env import DirectionPredictionEnv

try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = r"C:/Users/yassi/Documents/work"
os.makedirs(SAVE_DIR, exist_ok=True)
def spath(name): return os.path.join(SAVE_DIR, name)

DIR_NAMES  = ["North", "East", "South", "West"]
DIR_ARROWS = ["↑",     "→",    "↓",     "←"]

CFG = dict(
    place             = "Nördlingen, Germany",
    n_rsu             = 20, # Will be updated dynamically
    n_dir             = 4,
    state_dim         = 8,
    seq_len           = 60,
    trend_horizon     = 60,
    action_dim        = 4,
    hidden            = 256,
    tcn_channels      = [128, 256],
    attn_heads        = 4,
    dropout           = 0.10,
    rsu_min_dist      = 300,
    gnb_range         = 800,
    lr                = 2.5e-4,
    lr_min            = 1e-5,
    gamma             = 0.99,
    lam               = 0.95,
    clip_eps          = 0.20,
    vf_coef           = 0.5,
    ent_coef          = 0.10,
    max_grad          = 0.5,
    ppo_epochs        = 8,
    batch             = 512,
    rollout_steps     = 4096,
    total_steps       = 3_000_000,
    reward_correct    = 2.0,
    reward_wrong      = -1.0,
    data_steps        = 200_000,
    rsu_offsets       = [0, 4, 9, 16, 22, 28, 35, 42, 50, 58, 67, 76, 86, 96, 107, 118, 130, 142, 155, 168],
    cross_spillover   = 0.08,
    rush_drift_amp    = 45.0,
    rush_drift_period = 0.3,
    data_file         = spath("traffic_pred.npz"),
    log_every         = 20,
    warmup_frac       = 0.05,
    min_ep_len        = 2000,
    train_frac        = 0.80,
    log_100k_every    = 100_000,
    n_eval_samples    = 1000,
    lstm_hidden       = 128,
    lstm_layers       = 2,
    lstm_lr           = 1e-3,
    lstm_epochs       = 30,
    lstm_batch        = 512,
    arima_order       = (5, 1, 2),
    meta_online_window   = 50,
    meta_update_freq     = 3,
    meta_online_iters    = 25,
    pso_particles     = 40,
    pso_w             = 0.65,
    pso_c1            = 1.8,
    pso_c2            = 1.8,
    ga_pop_size       = 40,
    ga_crossover_rate = 0.80,
    ga_mutation_rate  = 0.20,
    ga_mutation_sigma = 0.10,
    ga_elite_frac     = 0.12,
    nonstat_seg_len   = 750,
    nonstat_regimes   = [8.0, 20.0, 2.0, 17.0],
    nonstat_file      = spath("nonstat_stream.npz"),
    latency_samples   = 200,
)

def fetch_intersections():
    print(f"Fetching road network around {CFG['place']}")
    G = ox.graph_from_address(CFG["place"], dist=1200, network_type="drive")
    nodes, _ = ox.graph_to_gdfs(G)
    nodes_3857 = nodes.to_crs(epsg=3857)
    
    gnb_x = nodes_3857.geometry.x.mean()
    gnb_y = nodes_3857.geometry.y.mean()
    
    dists = np.sqrt((nodes_3857.geometry.x - gnb_x)**2 + (nodes_3857.geometry.y - gnb_y)**2)
    in_range_nodes = nodes_3857[dists <= CFG["gnb_range"]]
    
    deg = dict(G.degree())
    in_range_ids = sorted(in_range_nodes.index, key=lambda n: deg.get(n, 0), reverse=True)
    
    sel_ids, sel_coords = [], []
    for nid in in_range_ids:
        if len(sel_ids) >= CFG["n_rsu"]: break
        x, y = nodes_3857.loc[nid, "geometry"].x, nodes_3857.loc[nid, "geometry"].y
        if all(np.sqrt((x-sx)**2 + (y-sy)**2) >= CFG["rsu_min_dist"] for sx, sy in sel_coords):
            sel_ids.append(nid); sel_coords.append((x, y))
            
    print(f"Found {len(sel_ids)} RSU locations ({CFG['rsu_min_dist']}m spacing)")
    # Return lat/lon for simulation
    final_coords = [(nodes.loc[nid, "y"], nodes.loc[nid, "x"]) for nid in sel_ids]
    return final_coords, G

def generate_dataset(coords, steps=200_000):
    T, R    = steps, len(coords)
    # Update CFG with the actual number of RSUs found
    CFG["n_rsu"] = R
    n_dir   = CFG["n_dir"]; H = CFG["trend_horizon"]; sd = CFG["state_dim"]
    cs      = CFG["cross_spillover"]; drift_a = CFG["rush_drift_amp"] / 60.0
    drift_p = CFG["rush_drift_period"]
    # Ensure offsets match the number of RSUs
    offsets = CFG["rsu_offsets"][:R]
    X = np.zeros((T, R, sd),    dtype=np.float32)
    D = np.zeros((T, R, n_dir), dtype=np.float32)
    # dir_phase[r,d] = hour-of-day offset for RSU r, direction d
    # Aligned with top2.py logic for predictable global trends
    base_dir = [0, 4, 9, 16]
    r_shift  = [0, 2, 1, 3]
    dir_phase = np.array([[(base_dir[d] + r_shift[r % 4]) % 24.0
                           for d in range(n_dir)] for r in range(R)], dtype=float)
    print(f"Generating {T:,} timesteps")
    base = np.zeros((T, R), dtype=np.float32)
    for t in range(T):
        hour  = (t / 60.0) % 24.0
        drift = drift_a * math.sin(2 * math.pi * t / (T * drift_p))
        for r in range(R):
            h_r  = (hour - offsets[r]) % 24.0
            rush = (math.exp(-((h_r-8-drift)**2)/4) + math.exp(-((h_r-17-drift)**2)/4))
            base[t, r] = np.clip(rush * 40 + np.random.normal(0, 5), 0, 80)
    spill = base.copy()
    for t in range(1, T):
        for r in range(R):
            spill[t, r] = np.clip(base[t,r] + cs*base[t-1,(r-1)%R] + cs*base[t-1,(r+1)%R], 0, 100)
    for t in range(T):
        hour  = (t / 60.0) % 24.0
        drift = drift_a * math.sin(2 * math.pi * t / (T * drift_p))
        for r in range(R):
            dens = float(spill[t, r])
            X[t, r] = [dens,
                np.clip(50-dens*0.4+np.random.normal(0,3),5,60),
                np.clip(dens*0.3+np.random.normal(0,2),0,30),
                np.clip(dens*0.15+np.random.normal(0,1),0,20),
                np.clip(0.01+dens*0.001+np.random.uniform(0,.02),0,.15),
                np.clip(1+dens*0.02+np.random.normal(0,.5),1,10),
                np.clip(dens/80+np.random.uniform(0,.1),0,1),
                math.sin(2*math.pi*hour/24.0)]
            for d in range(n_dir):
                h_d    = (hour - dir_phase[r,d] - drift) % 24.0
                rush_d = math.exp(-((h_d-8)**2)/5) + math.exp(-((h_d-17)**2)/5)
                D[t,r,d] = np.clip(rush_d*35+np.random.normal(0,4),0,70)
    Z = np.zeros(T, dtype=np.int64)
    for t in range(T - H):
        deltas = np.array([D[t+1:t+H+1,:,d].mean()-D[t,:,d].mean() for d in range(n_dir)])
        Z[t] = int(np.argmax(deltas))
    Z[T-H:] = Z[T-H-1]
    X = (X-X.mean(axis=(0,1),keepdims=True))/(X.std(axis=(0,1),keepdims=True).clip(1e-6))
    D = (D-D.mean(axis=(0,1),keepdims=True))/(D.std(axis=(0,1),keepdims=True).clip(1e-6))
    np.savez(CFG["data_file"], X=X, D=D, Z=Z)
    counts = Counter(Z.tolist())
    print(f"Dataset: X{X.shape}, D{D.shape}")
    print(f"  Labels: { {DIR_NAMES[k]: f'{counts[k]/T*100:.1f}%' for k in sorted(counts)} }")
    return X, D, Z

def generate_nonstationary_stream():
    seg_len = CFG["nonstat_seg_len"]; regimes = CFG["nonstat_regimes"]
    n_seg   = len(regimes); T_total = seg_len * n_seg
    R = CFG["n_rsu"]; n_dir = CFG["n_dir"]; sd = CFG["state_dim"]; H = CFG["trend_horizon"]
    # dir_phase[r,d] = hour-of-day offset for RSU r, direction d
    base_dir = [0, 4, 9, 16]
    r_shift  = [0, 2, 1, 3]
    dir_phase = np.array([[(base_dir[d] + r_shift[r % 4]) % 24.0
                           for d in range(n_dir)] for r in range(R)], dtype=float)
    Xs = np.zeros((T_total,R,sd), dtype=np.float32)
    Ds = np.zeros((T_total,R,n_dir), dtype=np.float32)
    Zs = np.zeros(T_total, dtype=np.int64)
    seg_bounds = []; global_t = 0
    print(f"\nGenerating non-stationary stream ({T_total} steps)")
    for seg_idx, peak_hour in enumerate(regimes):
        start = global_t
        for local_t in range(seg_len):
            hour = (peak_hour + local_t / 60.0) % 24.0
            for r in range(R):
                rush = (math.exp(-((hour-peak_hour)**2)/3)
                       + math.exp(-((hour-(peak_hour-9))%24)**2/8))
                dens = float(np.clip(rush*45+np.random.normal(0,6),0,90))
                Xs[global_t,r] = [dens,
                    np.clip(50-dens*0.4+np.random.normal(0,3),5,60),
                    np.clip(dens*0.3+np.random.normal(0,2),0,30),
                    np.clip(dens*0.15+np.random.normal(0,1),0,20),
                    np.clip(0.01+dens*0.001+np.random.uniform(0,.02),0,.15),
                    np.clip(1+dens*0.02+np.random.normal(0,.5),1,10),
                    np.clip(dens/90+np.random.uniform(0,.1),0,1),
                    math.sin(2*math.pi*hour/24.0)]
                for d in range(n_dir):
                    phase_shift = (seg_idx*3.5)%24.0
                    h_d = (hour-dir_phase[r,d]-phase_shift)%24.0
                    rush_d = (math.exp(-((h_d-peak_hour%12)**2)/4)
                             + math.exp(-((h_d-(peak_hour%12+6)%12)**2)/6))
                    Ds[global_t,r,d] = np.clip(rush_d*40+np.random.normal(0,5),0,75)
            global_t += 1
        for t in range(start, start+seg_len-H):
            deltas = np.array([Ds[t+1:t+H+1,:,d].mean()-Ds[t,:,d].mean() for d in range(n_dir)])
            Zs[t] = int(np.argmax(deltas))
        for t in range(start+seg_len-H, start+seg_len):
            Zs[t] = Zs[start+seg_len-H-1]
        seg_bounds.append((start, global_t))
    Xs = (Xs-Xs.mean(axis=(0,1),keepdims=True))/(Xs.std(axis=(0,1),keepdims=True).clip(1e-6))
    Ds = (Ds-Ds.mean(axis=(0,1),keepdims=True))/(Ds.std(axis=(0,1),keepdims=True).clip(1e-6))
    np.savez(CFG["nonstat_file"], Xs=Xs, Ds=Ds, Zs=Zs)
    regime_labels = ["Seg 0: Morning rush (08:00) — in-distribution",
                     "Seg 1: Weekend evening (20:00) — sudden shift",
                     "Seg 2: Late-night (02:00) — low traffic",
                     "Seg 3: Afternoon rush (17:00) — partial shift"]
    return Xs, Ds, Zs, seg_bounds, regime_labels

def make_splits(T, seq_len, trend_horizon, train_frac):
    valid = np.arange(seq_len, T - trend_horizon - 2)
    cut   = int(len(valid) * train_frac)
    return valid[:cut], valid[cut:]

def compute_class_weights(Z, train_idx, n_cls):
    counts  = np.bincount(Z[train_idx], minlength=n_cls).astype(float)
    total   = counts.sum()
    weights = total / (n_cls * counts.clip(1))
    return (weights / weights.mean()).astype(np.float32)

def make_gym_env(X, D, Z, train_idx, cfg, class_weights=None):
    """Create a Gymnasium-compliant DirectionPredictionEnv."""
    return DirectionPredictionEnv(
        X=X, D=D, Z=Z,
        valid_indices=train_idx,
        cfg=cfg,
        class_weights=class_weights,
    )

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_c, out_c, k, dilation=1):
        super().__init__(in_c, out_c, k, padding=(k-1)*dilation, dilation=dilation)
    def forward(self, x): return super().forward(x)[..., :x.size(-1)]

class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, dilation=1):
        super().__init__()
        self.net = nn.Sequential(CausalConv1d(in_c,out_c,k,dilation), nn.GELU(),
                                  CausalConv1d(out_c,out_c,k,dilation), nn.GELU())
        self.res = nn.Conv1d(in_c,out_c,1) if in_c!=out_c else nn.Identity()
    def forward(self, x): return self.net(x) + self.res(x)

class RSUEncoder(nn.Module):
    def __init__(self, in_dim, tcn_channels, out_dim, dropout=0.1):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv1d(in_dim,tcn_channels[0],3,padding=1), nn.GELU(),
            nn.Conv1d(tcn_channels[0],tcn_channels[0],3,padding=1), nn.GELU())
        ch, layers = tcn_channels[0], []
        for i, oc in enumerate(tcn_channels):
            layers.append(TCNBlock(ch,oc,dilation=2**i)); ch = oc
        self.tcn  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch,out_dim), nn.GELU())
    def forward(self, x):
        x = x.permute(0,2,1); x = self.entry(x); x = self.tcn(x)
        return self.proj(self.pool(x).squeeze(-1))

class DirectionActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_rsu  = cfg["n_rsu"]; self.n_dir = cfg["n_dir"]
        self.seq    = cfg["seq_len"]; self.in_dim = cfg["state_dim"]+cfg["n_dir"]
        enc_out = cfg["hidden"]; drop = cfg["dropout"]
        self.encoder   = RSUEncoder(self.in_dim, cfg["tcn_channels"], enc_out, dropout=drop)
        self.attn      = nn.MultiheadAttention(enc_out, cfg["attn_heads"], batch_first=True, dropout=drop)
        self.norm      = nn.LayerNorm(enc_out)
        self.global_fc = nn.Sequential(nn.Linear(enc_out,enc_out), nn.GELU(),
                                        nn.Dropout(drop), nn.Linear(enc_out,enc_out//2), nn.GELU())
        d = enc_out + enc_out//2
        self.actor  = nn.Sequential(nn.Linear(d,d//2), nn.GELU(), nn.Dropout(drop), nn.Linear(d//2,cfg["action_dim"]))
        self.critic = nn.Sequential(nn.Linear(d,d//2), nn.GELU(), nn.Linear(d//2,1))

    def _encode(self, flat):
        B, n, seq, d = flat.shape[0], self.n_rsu, self.seq, self.in_dim
        win = flat.view(B,seq,n,d).permute(0,2,1,3)
        emb = self.encoder(win.reshape(B*n,seq,d)).view(B,n,-1)
        att, _ = self.attn(emb,emb,emb); emb = self.norm(att+emb)
        agg = emb.mean(dim=1)
        return torch.cat([agg, self.global_fc(agg)], dim=-1)

    def forward(self, x):
        z = self._encode(x); return self.actor(z), self.critic(z).squeeze(-1)

    def act(self, x):
        logits, val = self.forward(x); dist = Categorical(logits=logits)
        a = dist.sample(); return a, dist.log_prob(a), val

    def evaluate(self, x, a):
        logits, val = self.forward(x); dist = Categorical(logits=logits)
        return dist.log_prob(a), dist.entropy(), val

class WarmupCosineScheduler:
    def __init__(self, opt, warmup, total, lr, lr_min):
        self.opt=opt; self.ws=warmup; self.ts=total; self.lr=lr; self.lrm=lr_min; self._s=0
    def step(self):
        self._s += 1; s = self._s
        if s < self.ws: scale = s/max(self.ws,1)
        else:
            p = (s-self.ws)/max(self.ts-self.ws,1)
            scale = self.lrm/self.lr+(1-self.lrm/self.lr)*0.5*(1+math.cos(math.pi*p))
        for pg in self.opt.param_groups: pg["lr"] = self.lr*scale
    def get_lr(self): return self.opt.param_groups[0]["lr"]

def compute_gae(rewards, values, dones, gamma, lam, last_value=0.0):
    """GAE with proper episode-boundary awareness and bootstrapping."""
    adv = np.zeros(len(rewards), dtype=np.float32)
    nv  = last_value
    g   = 0.0
    for i in reversed(range(len(rewards))):
        delta  = rewards[i] + gamma * (1 - dones[i]) * nv - values[i]
        g      = delta + gamma * lam * (1 - dones[i]) * g
        adv[i] = g
        nv     = values[i]
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns

class PPOTrainer:
    def __init__(self, model, cfg):
        self.model=model; self.cfg=cfg
        self.opt = optim.AdamW(model.parameters(), lr=cfg["lr"], eps=1e-5, weight_decay=1e-5)
        total=cfg["total_steps"]//cfg["rollout_steps"]; warmup=max(1,int(total*cfg["warmup_frac"]))
        self.sched = WarmupCosineScheduler(self.opt, warmup, total, cfg["lr"], cfg["lr_min"])

    def update(self, buf):
        states  = torch.FloatTensor(np.array(buf["s"])).to(DEVICE)
        actions = torch.LongTensor(buf["a"]).to(DEVICE)
        old_lp  = torch.FloatTensor(buf["lp"]).to(DEVICE)
        returns = torch.FloatTensor(buf["ret"]).to(DEVICE)
        advs    = torch.FloatTensor(buf["adv"]).to(DEVICE)
        advs    = (advs-advs.mean())/(advs.std()+1e-8)
        for _ in range(self.cfg["ppo_epochs"]):
            idx = np.random.permutation(len(states))
            for start in range(0, len(states), self.cfg["batch"]):
                b = idx[start:start+self.cfg["batch"]]
                lp, ent, val = self.model.evaluate(states[b], actions[b])
                ratio = (lp-old_lp[b]).exp()
                clip  = ratio.clamp(1-self.cfg["clip_eps"], 1+self.cfg["clip_eps"])
                loss  = (-torch.min(ratio*advs[b], clip*advs[b]).mean()
                         + self.cfg["vf_coef"]*(val-returns[b]).pow(2).mean()
                         - self.cfg["ent_coef"]*ent.mean())
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["max_grad"])
                self.opt.step()
        self.sched.step()

def train_ppo(env, model, trainer, cfg):
    BUF_KEYS = "s a lp v r d".split()
    buf = {k: [] for k in BUF_KEYS}
    rewards_log=[]; acc_log=[]
    ep_rew=ep=correct=ep_steps=0; best_avg=-np.inf
    next_100k=cfg["log_100k_every"]; step=0
    state, _info = env.reset()
    print(f"\nStarting PPO training, {cfg['total_steps']:,} steps")
    t_start = time.time()
    while step < cfg["total_steps"]:
        s_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): a, lp, v = model.act(s_t)
        ns, reward, terminated, truncated, info = env.step(a.item())
        done = terminated or truncated
        buf["s"].append(state.copy()); buf["a"].append(a.item())
        buf["lp"].append(lp.item());   buf["v"].append(v.item())
        buf["r"].append(reward)
        buf["d"].append(1.0 if terminated else 0.0)
        ep_rew+=reward; step+=1; ep_steps+=1; correct+=int(a.item()==info["true_dir"])
        if done:
            rewards_log.append(ep_rew); acc_log.append(correct/max(ep_steps,1))
            ep_rew=ep_steps=correct=0; ep+=1; state, _info = env.reset()
        else: state=ns
        if len(buf["r"]) >= cfg["rollout_steps"]:
            if done:
                last_val = 0.0
            else:
                with torch.no_grad():
                    s_last = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    _, last_val = model.forward(s_last)
                    last_val = last_val.item()
            adv, ret = compute_gae(
                buf["r"], buf["v"], buf["d"],
                cfg["gamma"], cfg["lam"], last_value=last_val)
            buf["adv"]=list(adv); buf["ret"]=list(ret)
            trainer.update(buf); buf={k:[] for k in BUF_KEYS}
            if rewards_log and ep%cfg["log_every"]==0:
                avg_r=np.mean(rewards_log[-20:]); avg_a=np.mean(acc_log[-20:])*100
                vram=(f" | VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB" if DEVICE.type=="cuda" else "")
                print(f"Step {step:9,d} | Ep {ep:5d} | Rew {avg_r:7.0f} | Acc {avg_a:5.1f}% | LR {trainer.sched.get_lr():.2e}{vram}")
            if step>=next_100k:
                avg_a=np.mean(acc_log[-50:])*100 if acc_log else 0
                print(f"  @ {step:,} ({step/cfg['total_steps']*100:.1f}%)  Acc={avg_a:.1f}%")
                next_100k+=cfg["log_100k_every"]
            if rewards_log:
                avg=np.mean(rewards_log[-20:])
                if avg>best_avg:
                    best_avg=avg; torch.save(model.state_dict(), spath("ppo_best.pt"))
    ppo_train_time = time.time()-t_start
    print(f"PPO finished in {ppo_train_time:.1f}s, best avg reward: {best_avg:.2f}")
    return rewards_log, acc_log, ppo_train_time

class OnlineMetaheuristicGuide:
    def __init__(self, n_dir, method="PSO", cfg=None):
        cfg=cfg or {}; self.n_dir=n_dir; self.method=method.upper()
        self.window_size=cfg.get("meta_online_window",50); self.update_freq=cfg.get("meta_update_freq",3)
        self.n_iter=cfg.get("meta_online_iters",25); self.n_pop=cfg.get("pso_particles",40)
        self.pso_w=cfg.get("pso_w",0.65); self.pso_c1=cfg.get("pso_c1",1.8); self.pso_c2=cfg.get("pso_c2",1.8)
        self.ga_cx=cfg.get("ga_crossover_rate",0.80); self.ga_mut=cfg.get("ga_mutation_rate",0.20)
        self.ga_sig=cfg.get("ga_mutation_sigma",0.10); self.n_elite=max(1,int(self.n_pop*cfg.get("ga_elite_frac",0.12)))
        self.bias=np.zeros(n_dir,dtype=np.float32); self._window=[]; self._step=0
        self._p_pos=np.random.uniform(-0.5,0.5,(self.n_pop,n_dir))
        self._p_vel=np.zeros_like(self._p_pos); self._p_best=self._p_pos.copy()
        self._p_bsc=np.zeros(self.n_pop); self._g_best=np.zeros(n_dir,dtype=np.float32); self._g_bsc=0.0
        self._ga_pop=np.random.uniform(-0.5,0.5,(self.n_pop,n_dir))
        self.running_correct=[]; self.bias_snapshots=[]; self.update_times_ms=[]

    def _score(self, bias):
        if not self._window: return 0.0
        return sum(int(np.argmax(l+bias)==lbl) for l,lbl in self._window)/len(self._window)

    def predict(self, raw_logits): return int(np.argmax(raw_logits+self.bias))

    def observe(self, raw_logits, true_label):
        self.running_correct.append(int(np.argmax(raw_logits+self.bias)==int(true_label)))
        self._window.append((raw_logits.copy(), int(true_label)))
        if len(self._window)>self.window_size: self._window.pop(0)
        self._step+=1
        if self._step%self.update_freq==0 and len(self._window)>=5:
            t0=time.perf_counter()
            if self.method=="PSO": self._pso_update()
            else: self._ga_update()
            self.update_times_ms.append((time.perf_counter()-t0)*1000)
            self.bias_snapshots.append(self.bias.copy())

    def _pso_update(self):
        n,d=self.n_pop,self.n_dir
        scores=np.array([self._score(p) for p in self._p_pos])
        improved=scores>self._p_bsc; self._p_best[improved]=self._p_pos[improved]; self._p_bsc[improved]=scores[improved]
        bi=int(np.argmax(self._p_bsc))
        if self._p_bsc[bi]>self._g_bsc: self._g_best=self._p_best[bi].copy(); self._g_bsc=self._p_bsc[bi]
        for _ in range(self.n_iter):
            r1,r2=np.random.rand(n,d),np.random.rand(n,d)
            self._p_vel=(self.pso_w*self._p_vel+self.pso_c1*r1*(self._p_best-self._p_pos)+self.pso_c2*r2*(self._g_best-self._p_pos))
            self._p_pos=np.clip(self._p_pos+self._p_vel,-3.,3.)
            scores=np.array([self._score(p) for p in self._p_pos])
            improved=scores>self._p_bsc; self._p_best[improved]=self._p_pos[improved]; self._p_bsc[improved]=scores[improved]
            bi=int(np.argmax(self._p_bsc))
            if self._p_bsc[bi]>self._g_bsc: self._g_best=self._p_best[bi].copy(); self._g_bsc=self._p_bsc[bi]
        self.bias=self._g_best.copy()

    def _ga_update(self):
        n,d=self.n_pop,self.n_dir
        for _ in range(self.n_iter):
            scores=np.array([self._score(p) for p in self._ga_pop])
            elite_idx=np.argsort(scores)[-self.n_elite:]; new_pop=list(self._ga_pop[elite_idx])
            while len(new_pop)<n:
                k=min(3,n); t1=np.random.choice(n,k,replace=False); t2=np.random.choice(n,k,replace=False)
                c1=int(t1[np.argmax(scores[t1])]); c2=int(t2[np.argmax(scores[t2])])
                child=(np.random.rand(d)*self._ga_pop[c1]+(1-np.random.rand(d))*self._ga_pop[c2]
                       if np.random.rand()<self.ga_cx else self._ga_pop[c1].copy())
                if np.random.rand()<self.ga_mut: child+=np.random.randn(d)*self.ga_sig
                new_pop.append(np.clip(child,-3.,3.))
            self._ga_pop=np.array(new_pop[:n])
        final=np.array([self._score(p) for p in self._ga_pop])
        self.bias=self._ga_pop[int(np.argmax(final))].copy()

    def rolling_accuracy(self, window=100):
        arr=np.array(self.running_correct,dtype=float)
        if len(arr)<window: return arr*100
        return np.convolve(arr,np.ones(window)/window,mode="valid")*100

def benchmark_latency(model, lstm_model, fitted_arima, X, D, Z, eval_idx, cfg):
    seq=cfg["seq_len"]; n_rsu=cfg["n_rsu"]; sd=cfg["state_dim"]
    n_dir=cfg["n_dir"]; H=cfg["trend_horizon"]; in_dim=n_rsu*(sd+n_dir)
    N=min(cfg["latency_samples"],len(eval_idx)); idx=eval_idx[:N]
    model.eval(); lstm_model.eval()

    guide_lat=OnlineMetaheuristicGuide(n_dir,"GA",cfg)
    t0=time.perf_counter()
    for t in idx:
        obs=np.concatenate([X[t-seq:t],D[t-seq:t]],axis=-1).flatten().astype(np.float32)
        with torch.no_grad(): logits,_=model.forward(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
        raw=logits.squeeze(0).cpu().numpy(); pred=guide_lat.predict(raw); guide_lat.observe(raw,int(Z[t]))
    ppo_ga_lat=(time.perf_counter()-t0)/N*1000

    t0=time.perf_counter()
    for t in idx:
        win=np.concatenate([X[t-seq:t],D[t-seq:t]],axis=-1).reshape(seq,in_dim).astype(np.float32)
        with torch.no_grad(): lstm_model(torch.FloatTensor(win).unsqueeze(0).to(DEVICE))
    lstm_lat=(time.perf_counter()-t0)/N*1000

    arima_times=[]
    for t in idx[:50]:
        t0=time.perf_counter()
        for d in range(n_dir):
            local=D[t-seq:t,:,d].mean(axis=1).astype(float)
            try:
                if fitted_arima and fitted_arima[d]: fitted_arima[d].apply(local,refit=False).forecast(H)
                else: raise ValueError()
            except Exception:
                slope=(local[-1]-local[0])/max(seq-1,1); _=local[-1]+slope*np.arange(1,H+1)
        arima_times.append((time.perf_counter()-t0)*1000)
    arima_lat=float(np.mean(arima_times))

    t0=time.perf_counter()
    for t in idx:
        for d in range(n_dir):
            _kalman_forecast(D[t-seq:t,:,d].mean(axis=1).astype(float),H)
    kalman_lat=(time.perf_counter()-t0)/N*1000

    t0=time.perf_counter()
    for _ in idx: np.random.randint(0,n_dir)
    random_lat=(time.perf_counter()-t0)/N*1000

    return dict(ppo_ga=ppo_ga_lat, lstm=lstm_lat, arima=arima_lat, kalman=kalman_lat, random=random_lat)

def eval_ppo_raw(model, X, D, test_idx, cfg):
    seq=cfg["seq_len"]; preds=np.zeros(len(test_idx),dtype=np.int64); model.eval()
    for i,t in enumerate(test_idx):
        obs=np.concatenate([X[t-seq:t],D[t-seq:t]],axis=-1).flatten().astype(np.float32)
        with torch.no_grad(): logits,_=model.forward(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)); preds[i]=logits.argmax(1).item()
    return preds

def eval_ppo_online_guided(model, X, D, Z, test_idx, cfg, guide):
    seq=cfg["seq_len"]; preds=np.zeros(len(test_idx),dtype=np.int64); model.eval()
    for i,t in enumerate(test_idx):
        obs=np.concatenate([X[t-seq:t],D[t-seq:t]],axis=-1).flatten().astype(np.float32)
        with torch.no_grad(): logits,_=model.forward(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
        raw=logits.squeeze(0).cpu().numpy(); preds[i]=guide.predict(raw); guide.observe(raw,int(Z[t]))
    return preds

def eval_nonstationary(model, lstm_model, Xs, Ds, Zs, seg_bounds, regime_labels, cfg):
    seq=cfg["seq_len"]; n_dir=cfg["n_dir"]; in_dim=cfg["n_rsu"]*(cfg["state_dim"]+cfg["n_dir"])
    guide_pso=OnlineMetaheuristicGuide(n_dir,"PSO",cfg); guide_ga=OnlineMetaheuristicGuide(n_dir,"GA",cfg)
    correct_pso=[]; correct_ga=[]; correct_lstm=[]
    seg_acc_pso=[]; seg_acc_ga=[]; seg_acc_lstm=[]
    print(f"\nRunning non-stationary evaluation"); model.eval(); lstm_model.eval()
    for si,(seg_start,seg_end) in enumerate(seg_bounds):
        print(f"  {regime_labels[si]}"); sp=[]; sg=[]; sl=[]
        for t in range(max(seg_start,seq),seg_end):
            obs=np.concatenate([Xs[t-seq:t],Ds[t-seq:t]],axis=-1).flatten().astype(np.float32); true=int(Zs[t])
            with torch.no_grad(): logits,_=model.forward(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
            raw=logits.squeeze(0).cpu().numpy()
            pred_pso=guide_pso.predict(raw); guide_pso.observe(raw,true); sp.append(int(pred_pso==true))
            pred_ga=guide_ga.predict(raw);  guide_ga.observe(raw,true);  sg.append(int(pred_ga==true))
            win=np.concatenate([Xs[t-seq:t],Ds[t-seq:t]],axis=-1).reshape(seq,in_dim).astype(np.float32)
            with torch.no_grad(): pred_lstm=lstm_model(torch.FloatTensor(win).unsqueeze(0).to(DEVICE)).argmax(1).item()
            sl.append(int(pred_lstm==true))
        seg_acc_pso.append(np.mean(sp)*100); seg_acc_ga.append(np.mean(sg)*100); seg_acc_lstm.append(np.mean(sl)*100)
        correct_pso.extend(sp); correct_ga.extend(sg); correct_lstm.extend(sl)
        print(f"    PSO {seg_acc_pso[-1]:.1f}%  GA {seg_acc_ga[-1]:.1f}%  LSTM {seg_acc_lstm[-1]:.1f}%")
    return dict(correct_pso=np.array(correct_pso,dtype=float), correct_ga=np.array(correct_ga,dtype=float),
                correct_lstm=np.array(correct_lstm,dtype=float), seg_acc_pso=seg_acc_pso,
                seg_acc_ga=seg_acc_ga, seg_acc_lstm=seg_acc_lstm,
                overall_pso=float(np.mean(correct_pso))*100, overall_ga=float(np.mean(correct_ga))*100,
                overall_lstm=float(np.mean(correct_lstm))*100, guide_pso=guide_pso, guide_ga=guide_ga)

class LSTMBaseline(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, n_cls, dropout=0.3):
        super().__init__()
        self.lstm=nn.LSTM(in_dim,hidden,n_layers,batch_first=True,dropout=dropout,bidirectional=True)
        self.norm=nn.LayerNorm(hidden*2)
        self.head=nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,n_cls))
    def forward(self, x): out,_=self.lstm(x); return self.head(self.norm(out[:,-1,:]))

def train_lstm(X, D, Z, train_idx, cfg):
    seq=cfg["seq_len"]; n_rsu=cfg["n_rsu"]; in_dim=n_rsu*(cfg["state_dim"]+cfg["n_dir"])
    windows=np.array([np.concatenate([X[t-seq:t],D[t-seq:t]],axis=-1).reshape(seq,in_dim) for t in train_idx])
    labels=Z[train_idx]; W=torch.FloatTensor(windows).to(DEVICE); L=torch.LongTensor(labels).to(DEVICE)
    model=LSTMBaseline(in_dim,cfg["lstm_hidden"],cfg["lstm_layers"],cfg["n_dir"]).to(DEVICE)
    opt=optim.AdamW(model.parameters(),lr=cfg["lstm_lr"],weight_decay=1e-4)
    sched=optim.lr_scheduler.CosineAnnealingLR(opt,cfg["lstm_epochs"])
    n,bs=len(train_idx),cfg["lstm_batch"]; print(f"\nTraining LSTM on {n:,} windows")
    t0=time.time()
    for ep in range(cfg["lstm_epochs"]):
        model.train(); perm=torch.randperm(n); loss_tot=correct=0
        for s in range(0,n,bs):
            b=perm[s:s+bs]; logits=model(W[b]); loss=F.cross_entropy(logits,L[b])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_tot+=loss.item()*len(b); correct+=(logits.argmax(1)==L[b]).sum().item()
        sched.step()
        if (ep+1)%5==0: print(f"  Ep {ep+1:3d}  loss={loss_tot/n:.4f}  acc={correct/n*100:.1f}%")
    lstm_train_time=time.time()-t0; print(f"  LSTM done in {lstm_train_time:.1f}s")
    return model, lstm_train_time

def eval_lstm(model, X, D, test_idx, cfg):
    seq=cfg["seq_len"]; n_rsu=cfg["n_rsu"]; in_dim=n_rsu*(cfg["state_dim"]+cfg["n_dir"])
    wins=np.array([np.concatenate([X[t-seq:t],D[t-seq:t]],axis=-1).reshape(seq,in_dim) for t in test_idx])
    W=torch.FloatTensor(wins).to(DEVICE); preds=np.zeros(len(test_idx),dtype=np.int64); model.eval()
    for s in range(0,len(test_idx),256):
        with torch.no_grad(): preds[s:s+256]=model(W[s:s+256]).argmax(1).cpu().numpy()
    return preds

def fit_arima(D, train_idx, cfg):
    if not HAS_STATSMODELS: return None
    fitted,ts=[],np.sort(train_idx); print("\nFitting ARIMA models")
    for d in range(cfg["n_dir"]):
        series=D[ts,:,d].mean(axis=1).astype(float)
        try:
            res=_ARIMA(series,order=cfg["arima_order"],enforce_stationarity=False,enforce_invertibility=False).fit()
            fitted.append(res)
        except Exception: fitted.append(None)
    return fitted

def eval_arima(D, test_idx, fitted, cfg):
    if fitted is None: return np.random.randint(0,cfg["n_dir"],len(test_idx))
    seq,H,n_dir=cfg["seq_len"],cfg["trend_horizon"],cfg["n_dir"]
    preds=np.zeros(len(test_idx),dtype=np.int64)
    for i,t in enumerate(test_idx):
        deltas=np.zeros(n_dir)
        for d in range(n_dir):
            local=D[t-seq:t,:,d].mean(axis=1).astype(float); curr=float(D[t-1,:,d].mean())
            try:
                fc=(fitted[d].apply(local,refit=False).forecast(H) if fitted[d] else None)
                if fc is None: raise ValueError()
            except Exception:
                try: fc=_ARIMA(local,order=cfg["arima_order"],enforce_stationarity=False,enforce_invertibility=False).fit().forecast(H)
                except Exception: slope=(local[-1]-local[0])/max(seq-1,1); fc=local[-1]+slope*np.arange(1,H+1)
            deltas[d]=float(np.mean(fc))-curr
        preds[i]=int(np.argmax(deltas))
    return preds

def _kalman_forecast(series, H=60, damping=0.97):
    F_=np.array([[1.,1.],[0.,damping]]); H_=np.array([[1.,0.]])
    Q=np.array([[0.0125,0.025],[0.025,0.05]]); R_=np.array([[4.0]])
    x=np.array([series[0],0.]); P=np.eye(2)*10.
    for obs in series:
        x=F_@x; P=F_@P@F_.T+Q; y=obs-(H_@x)[0]; S=H_@P@H_.T+R_
        K=P@H_.T@np.linalg.inv(S); x=x+(K@np.array([[y]])).flatten(); P=(np.eye(2)-K@H_)@P
    fc=[]
    for _ in range(H): x=F_@x; fc.append(float(x[0]))
    return float(np.mean(fc))

def eval_kalman(D, test_idx, cfg):
    seq,H,n_dir=cfg["seq_len"],cfg["trend_horizon"],cfg["n_dir"]
    preds=np.zeros(len(test_idx),dtype=np.int64)
    for i,t in enumerate(test_idx):
        deltas=np.zeros(n_dir)
        for d in range(n_dir):
            deltas[d]=_kalman_forecast(D[t-seq:t,:,d].mean(axis=1).astype(float),H)-float(D[t-1,:,d].mean())
        preds[i]=int(np.argmax(deltas))
    return preds

def compute_metrics(preds, labels, n_cls=4):
    preds,labels=np.array(preds),np.array(labels); acc=float((preds==labels).mean())
    per_f1=np.zeros(n_cls); per_acc=np.zeros(n_cls)
    for c in range(n_cls):
        mask=labels==c
        if not mask.any(): continue
        tp=float(((preds==c)&(labels==c)).sum()); fp=float(((preds==c)&(labels!=c)).sum()); fn=float(((preds!=c)&(labels==c)).sum())
        per_acc[c]=tp/mask.sum(); p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); per_f1[c]=2*p*r/(p+r+1e-9)
    return dict(accuracy=acc, macro_f1=float(per_f1.mean()), per_class_acc=per_acc)

def plot_results(rewards_log, acc_log, results,
                 guide_pso, guide_ga,
                 ns_results, seg_bounds, regime_labels,
                 latency_ms, ppo_train_time, lstm_train_time, cfg):

    METHOD_COLORS = {"PPO+PSO":"#58a6ff","PPO+GA":"#3fb950","LSTM (supervised)":"#ffa657",
                     "ARIMA":"#d2a8ff","Kalman":"#e3b341","Random":"#8b949e"}
    DISPLAY_METHODS = [m for m in results.keys() if m != "PPO (raw)"]
    n_dir = cfg["n_dir"]

    fig = plt.figure(figsize=(26, 22), facecolor="#0d1117")
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.52, wspace=0.37)

    def _style(ax, title):
        ax.set_facecolor("#161b22"); ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.tick_params(colors="#8b949e", labelsize=8); ax.spines[:].set_color("#30363d"); ax.grid(alpha=.15, color="#30363d")

    lstm_acc_pct = results["LSTM (supervised)"]["accuracy"] * 100

    # Row 0
    ax1 = fig.add_subplot(gs[0, 0])
    if len(rewards_log) > 10:
        sm = np.convolve(rewards_log, np.ones(10)/10, mode="valid")
        ax1.plot(rewards_log, alpha=0.15, color="#58a6ff", lw=1)
        ax1.plot(sm, color="#58a6ff", lw=2, label="Smoothed (10ep)")
    ax1.set_xlabel("Episode", color="#8b949e", fontsize=8); ax1.set_ylabel("Cumulative Reward", color="#8b949e", fontsize=8)
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=7)
    _style(ax1, "Stage 1 — PPO Training Reward")

    ax2 = fig.add_subplot(gs[0, 1])
    if len(acc_log) > 10:
        sm2 = np.convolve(acc_log, np.ones(10)/10, mode="valid") * 100
        ax2.plot(np.array(acc_log)*100, alpha=0.15, color="#3fb950", lw=1)
        ax2.plot(sm2, color="#3fb950", lw=2, label="Smoothed (10ep)")
    ax2.axhline(25, color="#555", lw=1.2, ls="--", label="Random 25%"); ax2.set_ylim(0, 105)
    ax2.set_xlabel("Episode", color="#8b949e", fontsize=8); ax2.set_ylabel("Direction Accuracy (%)", color="#8b949e", fontsize=8)
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=7)
    _style(ax2, "Stage 1 — PPO Training Accuracy")

    ax3 = fig.add_subplot(gs[0, 2])
    methods=DISPLAY_METHODS; acc_vals=[results[m]["accuracy"]*100 for m in methods]; f1_vals=[results[m]["macro_f1"]*100 for m in methods]
    bc=[METHOD_COLORS.get(m,"#aaa") for m in methods]; x,w=np.arange(len(methods)),0.38
    b1=ax3.bar(x-w/2,acc_vals,w,color=bc,alpha=0.85,edgecolor="#0d1117",lw=0.6,label="Accuracy")
    ax3.bar(x+w/2,f1_vals,w,color=bc,alpha=0.45,edgecolor="#0d1117",lw=0.6,hatch="//",label="Macro-F1")
    for bar,v in zip(b1,acc_vals):
        ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.8,f"{v:.1f}",ha="center",va="bottom",color="white",fontsize=7,fontweight="bold")
    ax3.axhline(25,color="#555",lw=1.0,ls="--",alpha=0.7); ax3.set_xticks(x)
    ax3.set_xticklabels(methods,fontsize=7,color="white",rotation=20,ha="right"); ax3.set_ylim(0,115)
    ax3.set_ylabel("Score (%)",color="#8b949e",fontsize=8)
    ax3.legend(facecolor="#161b22",edgecolor="#30363d",labelcolor="white",fontsize=7)
    _style(ax3, "Static Test — All Methods Accuracy & Macro-F1")

    ax4 = fig.add_subplot(gs[0, 3])
    lat_labels=["PPO+GA","LSTM","ARIMA","Kalman","Random"]
    lat_vals=[latency_ms["ppo_ga"],latency_ms["lstm"],latency_ms["arima"],latency_ms["kalman"],latency_ms["random"]]
    lat_colors=["#3fb950","#ffa657","#d2a8ff","#e3b341","#8b949e"]
    bars4=ax4.bar(lat_labels,lat_vals,color=lat_colors,alpha=0.85,edgecolor="#0d1117",lw=0.6)
    for bar,v in zip(bars4,lat_vals):
        ax4.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.05,f"{v:.2f}ms",ha="center",va="bottom",color="white",fontsize=8,fontweight="bold")
    ax4.set_ylabel("Mean latency per sample (ms)",color="#8b949e",fontsize=8); ax4.set_ylim(0,max(lat_vals)*1.25)
    _style(ax4, "Inference Latency — Mean per Sample (ms)")

    # Row 1
    ax5 = fig.add_subplot(gs[1, 0])
    x_cls=np.arange(n_dir); bw5=0.13
    for k,m in enumerate(methods):
        pca=results[m]["per_class_acc"]*100; off=(k-len(methods)/2+0.5)*bw5
        ax5.bar(x_cls+off,pca,bw5,color=METHOD_COLORS.get(m,"#aaa"),alpha=0.85,label=m,edgecolor="#0d1117",lw=0.5)
    ax5.set_xticks(x_cls); ax5.set_xticklabels(DIR_NAMES,color="white",fontsize=9)
    ax5.axhline(25,color="#555",lw=1.0,ls="--",alpha=0.7); ax5.set_ylim(0,115)
    ax5.set_ylabel("Accuracy (%)",color="#8b949e",fontsize=8)
    ax5.legend(facecolor="#161b22",edgecolor="#30363d",labelcolor="white",fontsize=6,ncol=2)
    _style(ax5, "Static Test — Per-Direction Accuracy")

    ax6 = fig.add_subplot(gs[1, 1])
    thr_vals=[1000/v if v>0 else 0 for v in lat_vals]
    bars6=ax6.bar(lat_labels,thr_vals,color=lat_colors,alpha=0.85,edgecolor="#0d1117",lw=0.6)
    for bar,v in zip(bars6,thr_vals):
        lbl=f"{v/1e6:.1f}M/s" if v>1e6 else f"{v/1e3:.1f}k/s" if v>1e3 else f"{v:.0f}/s"
        ax6.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.02,lbl,ha="center",va="bottom",color="white",fontsize=7,fontweight="bold")
    ax6.set_ylabel("Samples / second",color="#8b949e",fontsize=8); ax6.set_ylim(0,max(thr_vals)*1.25)
    _style(ax6, "Throughput — Samples per Second (higher = better)")

    ax7 = fig.add_subplot(gs[1, 2])
    pso_ut=guide_pso.update_times_ms if guide_pso.update_times_ms else [0]
    ga_ut=guide_ga.update_times_ms  if guide_ga.update_times_ms  else [0]
    if len(pso_ut)>1:
        ax7.plot(pso_ut,color="#58a6ff",lw=1.2,alpha=0.6,label="PSO updates")
        ax7.axhline(np.mean(pso_ut),color="#58a6ff",lw=2,label=f"PSO mean {np.mean(pso_ut):.1f}ms")
    if len(ga_ut)>1:
        ax7.plot(ga_ut,color="#3fb950",lw=1.2,alpha=0.6,ls="--",label="GA updates")
        ax7.axhline(np.mean(ga_ut),color="#3fb950",lw=2,ls="--",label=f"GA mean {np.mean(ga_ut):.1f}ms")
    ax7.set_xlabel("Optimisation update #",color="#8b949e",fontsize=8); ax7.set_ylabel("Wall time (ms)",color="#8b949e",fontsize=8)
    ax7.legend(facecolor="#161b22",edgecolor="#30363d",labelcolor="white",fontsize=7)
    _style(ax7, "PSO / GA Per-Update Improvement Time (ms)")

    ax8 = fig.add_subplot(gs[1, 3])
    pso_total=sum(guide_pso.update_times_ms)/1000 if guide_pso.update_times_ms else 0
    ga_total=sum(guide_ga.update_times_ms)/1000   if guide_ga.update_times_ms  else 0
    nn_labels=["PPO\n(train)","LSTM\n(train)","PSO total\nopt time","GA total\nopt time"]
    nn_vals=[ppo_train_time,lstm_train_time,pso_total,ga_total]
    nn_colors=["#58a6ff","#ffa657","#58a6ff","#3fb950"]; nn_alpha=[0.85,0.85,0.45,0.45]
    for i,(v,c,a) in enumerate(zip(nn_vals,nn_colors,nn_alpha)):
        ax8.bar(i,v,color=c,alpha=a,edgecolor="#0d1117",lw=0.6)
        lbl=f"{v:.1f}s" if v<3600 else f"{v/3600:.1f}h"
        ax8.text(i,v+max(nn_vals)*0.01,lbl,ha="center",va="bottom",color="white",fontsize=8,fontweight="bold")
    ax8.set_xticks(range(len(nn_labels))); ax8.set_xticklabels(nn_labels,color="white",fontsize=8)
    ax8.set_ylabel("Wall time (s)",color="#8b949e",fontsize=8); ax8.set_ylim(0,max(nn_vals)*1.22)
    _style(ax8, "Neural Network & Optimiser Training Time (s)")

    # Row 2
    ax9 = fig.add_subplot(gs[2, 0:2])
    roll_w_ns=50
    ns_pso_roll =np.convolve(ns_results["correct_pso"], np.ones(roll_w_ns)/roll_w_ns,mode="valid")*100
    ns_ga_roll  =np.convolve(ns_results["correct_ga"],  np.ones(roll_w_ns)/roll_w_ns,mode="valid")*100
    ns_lstm_roll=np.convolve(ns_results["correct_lstm"],np.ones(roll_w_ns)/roll_w_ns,mode="valid")*100
    xs_roll=np.arange(len(ns_pso_roll))
    ax9.plot(xs_roll,ns_pso_roll, color="#58a6ff",lw=2.0,label="PPO+PSO (online adapt)")
    ax9.plot(xs_roll,ns_ga_roll,  color="#3fb950",lw=2.0,ls="--",label="PPO+GA (online adapt)")
    ax9.plot(xs_roll,ns_lstm_roll,color="#ffa657",lw=1.5,ls=":",label="LSTM (frozen — no adapt)")
    seg_names=["Seg 0\nMorning","Seg 1\nEvening","Seg 2\nLate-night","Seg 3\nAfternoon"]
    seg_cols=["#58a6ff","#3fb950","#ffa657","#d2a8ff"]
    for si,(s,e) in enumerate(seg_bounds):
        s_p=max(0,s-cfg["seq_len"]); e_p=max(0,e-cfg["seq_len"]-roll_w_ns+1)
        if e_p>s_p: ax9.axvspan(s_p,min(e_p,len(xs_roll)-1),alpha=0.10,color=seg_cols[si])
        mid=(s_p+min(e_p,len(xs_roll)-1))//2; ax9.axvline(s_p,color="#30363d",lw=1.0,ls="--",alpha=0.5)
        if mid<len(xs_roll): ax9.text(mid,97,seg_names[si],ha="center",va="top",color="#8b949e",fontsize=7)
    ax9.axhline(25,color="#555",lw=1.0,ls="--",alpha=0.5,label="Random")
    ax9.set_xlabel(f"Test step (rolling {roll_w_ns})",color="#8b949e",fontsize=8); ax9.set_ylabel("Accuracy (%)",color="#8b949e",fontsize=8); ax9.set_ylim(0,105)
    ax9.legend(facecolor="#161b22",edgecolor="#30363d",labelcolor="white",fontsize=8,loc="lower right")
    _style(ax9,"NON-STATIONARY TEST — Rolling Accuracy across 4 Shifting Traffic Regimes\nPPO+PSO/GA adapt online  |  LSTM frozen, cannot recover")

    ax10 = fig.add_subplot(gs[2, 2])
    n_segs=len(seg_bounds); x_segs=np.arange(n_segs); bw_seg=0.25
    ax10.bar(x_segs-bw_seg,ns_results["seg_acc_pso"], bw_seg,color="#58a6ff",alpha=0.85,edgecolor="#0d1117",label="PPO+PSO")
    ax10.bar(x_segs,        ns_results["seg_acc_ga"],  bw_seg,color="#3fb950",alpha=0.85,edgecolor="#0d1117",label="PPO+GA")
    ax10.bar(x_segs+bw_seg, ns_results["seg_acc_lstm"],bw_seg,color="#ffa657",alpha=0.85,edgecolor="#0d1117",label="LSTM")
    for si in range(n_segs):
        best_rl=max(ns_results["seg_acc_pso"][si],ns_results["seg_acc_ga"][si]); gap=best_rl-ns_results["seg_acc_lstm"][si]
        col="#3fb950" if gap>0 else "#ff7b72"
        ax10.text(x_segs[si],max(best_rl,ns_results["seg_acc_lstm"][si])+1.5,f"{gap:+.1f}%",ha="center",va="bottom",color=col,fontsize=8,fontweight="bold")
    ax10.set_xticks(x_segs); ax10.set_xticklabels([f"Seg {i}" for i in range(n_segs)],color="white",fontsize=8)
    ax10.set_ylim(0,110); ax10.set_ylabel("Accuracy (%)",color="#8b949e",fontsize=8); ax10.axhline(25,color="#555",lw=1.0,ls="--",alpha=0.6)
    ax10.legend(facecolor="#161b22",edgecolor="#30363d",labelcolor="white",fontsize=7)
    _style(ax10,"Non-Stationary — Per-Segment Accuracy\n(label = best RL − LSTM)")

    ax11 = fig.add_subplot(gs[2, 3])
    ns_m=["PPO+PSO\n(online)","PPO+GA\n(online)","LSTM\n(frozen)"]
    ns_a=[ns_results["overall_pso"],ns_results["overall_ga"],ns_results["overall_lstm"]]
    ns_c=["#58a6ff","#3fb950","#ffa657"]
    bars11=ax11.bar(ns_m,ns_a,color=ns_c,alpha=0.85,edgecolor="#0d1117",lw=0.6)
    for bar,v in zip(bars11,ns_a):
        ax11.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f"{v:.1f}%",ha="center",va="bottom",color="white",fontsize=10,fontweight="bold")
    ax11.axhline(25,color="#555",lw=1.2,ls="--",alpha=0.7,label="Random"); ax11.set_ylim(0,100)
    ax11.set_ylabel("Overall Accuracy (%)",color="#8b949e",fontsize=8)
    ax11.legend(facecolor="#161b22",edgecolor="#30363d",labelcolor="white",fontsize=7)
    _style(ax11,"Non-Stationary — Overall Accuracy\n(all 4 regimes combined)")

    # Row 3
    ax12 = fig.add_subplot(gs[3, 0])
    ns_pso_guide=ns_results["guide_pso"]
    pso_snaps=(np.array(ns_pso_guide.bias_snapshots) if ns_pso_guide.bias_snapshots else np.zeros((2,n_dir)))
    vabs=max(0.05,np.abs(pso_snaps).max())
    im12=ax12.imshow(pso_snaps.T,aspect="auto",cmap="RdBu",vmin=-vabs,vmax=vabs,interpolation="nearest")
    ax12.set_yticks(range(n_dir)); ax12.set_yticklabels(DIR_NAMES,color="white",fontsize=8)
    ax12.set_xlabel("Optimisation update #",color="#8b949e",fontsize=8)
    plt.colorbar(im12,ax=ax12,fraction=0.046,pad=0.04)
    _style(ax12,"PSO Bias Evolution — Non-Stationary Stream")

    ax13 = fig.add_subplot(gs[3, 1])
    scatter_methods=["PPO+GA","LSTM (supervised)","Kalman","Random"]
    scatter_lat=[latency_ms["ppo_ga"],latency_ms["lstm"],latency_ms["kalman"],latency_ms["random"]]
    scatter_acc=[results[m]["accuracy"]*100 for m in scatter_methods]
    scatter_cols=["#3fb950","#ffa657","#e3b341","#8b949e"]
    for m,lx,ay,sc in zip(scatter_methods,scatter_lat,scatter_acc,scatter_cols):
        ax13.scatter(lx,ay,s=120,color=sc,zorder=5,edgecolors="#0d1117",lw=1.2)
        ax13.annotate(m.replace(" (supervised)",""),[lx,ay],textcoords="offset points",xytext=[6,4],color="white",fontsize=7)
    ax13.set_xlabel("Mean Latency (ms)",color="#8b949e",fontsize=8); ax13.set_ylabel("Static Accuracy (%)",color="#8b949e",fontsize=8)
    _style(ax13,"Latency vs Accuracy Trade-off")

    ax14 = fig.add_subplot(gs[3, 2:4])
    ax14.axis("off")
    tbl_methods=["PPO+GA","LSTM (supervised)","ARIMA","Kalman","Random"]
    lat_map={"PPO+GA":latency_ms["ppo_ga"],"LSTM (supervised)":latency_ms["lstm"],
             "ARIMA":latency_ms["arima"],"Kalman":latency_ms["kalman"],"Random":latency_ms["random"]}
    rows14=[]
    for m in tbl_methods:
        r=results[m]; lat=lat_map[m]; thr=1000/lat if lat>0 else float("inf")
        thr_s=f"{thr/1e6:.2f}M/s" if thr>1e6 else f"{thr/1e3:.1f}k/s" if thr>1e3 else f"{thr:.0f}/s"
        rows14.append([m,f"{r['accuracy']*100:.2f}%",f"{r['macro_f1']*100:.2f}%",f"{lat:.3f}ms",thr_s])
    tbl14=ax14.table(cellText=rows14,colLabels=["Method","Accuracy","Macro-F1","Latency","Throughput"],loc="center",cellLoc="center")
    tbl14.auto_set_font_size(False); tbl14.set_fontsize(8); tbl14.scale(1,1.7)
    for (row,col),cell in tbl14.get_celld().items():
        cell.set_facecolor("#161b22" if row>0 else "#21262d"); cell.set_text_props(color="white"); cell.set_edgecolor("#30363d")
    _style(ax14,"Full Comparison — Accuracy · Latency · Throughput")

    fig.suptitle("V2X Direction Trend Prediction  ·  Stage 1: PPO (pure RL)  →  Stage 2: PSO/GA online logit-bias guidance\n"
                 "Top 2 rows: static test + latency/throughput  |  Bottom 2 rows: non-stationary test — 4 abrupt traffic regime shifts",
                 color="white",fontsize=11,y=0.998)

    out=spath("v2x_pred_ppo_online_meta.png")
    plt.savefig(out,dpi=160,bbox_inches="tight",facecolor=fig.get_facecolor())
    print(f"Plot saved to {out}")
    plt.show()

if __name__ == "__main__":


    coords, G = fetch_intersections()

    if os.path.exists(CFG["data_file"]):
        print("Loading cached dataset")
        d_=np.load(CFG["data_file"]); X,D,Z=d_["X"],d_["D"],d_["Z"]
        CFG["n_rsu"] = X.shape[1]
    else:
        X,D,Z=generate_dataset(coords,steps=CFG["data_steps"])
        CFG["n_rsu"] = X.shape[1]
    T=X.shape[0]

    train_idx,test_idx=make_splits(T,CFG["seq_len"],CFG["trend_horizon"],CFG["train_frac"])
    print(f"Split: {len(train_idx):,} train | {len(test_idx):,} test")

    rng=np.random.default_rng(42)
    s_eval=int(rng.integers(0,max(1,len(test_idx)-CFG["n_eval_samples"]-1)))
    eval_idx=test_idx[s_eval:s_eval+CFG["n_eval_samples"]]; eval_lbl=Z[eval_idx]

    class_weights=compute_class_weights(Z,train_idx,CFG["n_dir"])
    env=make_gym_env(X,D,Z,train_idx,CFG,class_weights=class_weights)

    model=DirectionActorCritic(CFG).to(DEVICE)
    trainer=PPOTrainer(model,CFG)


    rewards_log,acc_log,ppo_train_time=train_ppo(env,model,trainer,CFG)
    torch.save(model.state_dict(),spath("ppo_pred_final.pt"))

    lstm_model,lstm_train_time=train_lstm(X,D,Z,train_idx,CFG)
    fitted_arima=fit_arima(D,train_idx,CFG)

    ppo_raw_preds=eval_ppo_raw(model,X,D,eval_idx,CFG)

    print("\nEvaluating PPO+PSO")
    guide_pso=OnlineMetaheuristicGuide(CFG["n_dir"],"PSO",CFG)
    ppo_pso_preds=eval_ppo_online_guided(model,X,D,Z,eval_idx,CFG,guide_pso)

    print("Evaluating PPO+GA")
    guide_ga=OnlineMetaheuristicGuide(CFG["n_dir"],"GA",CFG)
    ppo_ga_preds=eval_ppo_online_guided(model,X,D,Z,eval_idx,CFG,guide_ga)

    lstm_preds=eval_lstm(lstm_model,X,D,eval_idx,CFG)
    arima_preds=eval_arima(D,eval_idx,fitted_arima,CFG)
    kalman_preds=eval_kalman(D,eval_idx,CFG)
    rnd_preds=rng.integers(0,CFG["n_dir"],len(eval_idx))

    results={"PPO+PSO":compute_metrics(ppo_pso_preds,eval_lbl),
             "PPO+GA":compute_metrics(ppo_ga_preds,eval_lbl),
             "LSTM (supervised)":compute_metrics(lstm_preds,eval_lbl),
             "ARIMA":compute_metrics(arima_preds,eval_lbl),
             "Kalman":compute_metrics(kalman_preds,eval_lbl),
             "Random":compute_metrics(rnd_preds,eval_lbl),
             "PPO (raw)":compute_metrics(ppo_raw_preds,eval_lbl)}

    lstm_acc_pct=results["LSTM (supervised)"]["accuracy"]*100
    DISPLAY_METHODS=[m for m in results.keys() if m!="PPO (raw)"]
    print("\nStatic test results:")
    print(f"  {'Method':<18s}  {'Accuracy':>8s}  {'Macro-F1':>8s}  {'vsLSTM':>7s}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*7}")
    for m in DISPLAY_METHODS:
        r=results[m]; delta=r["accuracy"]*100-lstm_acc_pct
        print(f"  {m:<18s}  {r['accuracy']*100:>7.2f}%  {r['macro_f1']*100:>7.2f}%  {delta:>+.2f}%")

    print("\nRunning latency benchmark")
    latency_ms=benchmark_latency(model,lstm_model,fitted_arima,X,D,Z,eval_idx,CFG)
    for k,v in latency_ms.items(): print(f"  {k:<10}: {v:.3f}ms  ({1000/v:.0f} samples/s)")

    if os.path.exists(CFG["nonstat_file"]):
        print("\nLoading cached non-stationary stream")
        ns=np.load(CFG["nonstat_file"]); Xs,Ds,Zs=ns["Xs"],ns["Ds"],ns["Zs"]
        seg_len=CFG["nonstat_seg_len"]
        seg_bounds=[(i*seg_len,(i+1)*seg_len) for i in range(len(CFG["nonstat_regimes"]))]
    else:
        Xs,Ds,Zs,seg_bounds,_=generate_nonstationary_stream()

    regime_labels=["Seg 0: Morning rush (08:00) — in-distribution",
                   "Seg 1: Weekend evening (20:00) — sudden shift",
                   "Seg 2: Late-night (02:00) — low traffic",
                   "Seg 3: Afternoon rush (17:00) — partial shift"]

    ns_results=eval_nonstationary(model,lstm_model,Xs,Ds,Zs,seg_bounds,regime_labels,CFG)

    print("\nNon-stationary test results:")
    print(f"  {'Regime':<16s}  {'PPO+PSO':>7s}  {'PPO+GA':>7s}  {'LSTM':>7s}  {'Gain':>6s}")
    print(f"  {'-'*16}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
    seg_short=["Morning","Evening","Late-night","Afternoon"]
    for si in range(len(seg_bounds)):
        pa=ns_results["seg_acc_pso"][si]; ga=ns_results["seg_acc_ga"][si]
        la=ns_results["seg_acc_lstm"][si]; gain=max(pa,ga)-la
        print(f"  {seg_short[si]:<16s}  {pa:>6.1f}%  {ga:>6.1f}%  {la:>6.1f}%  {gain:>+.1f}%")
    op=ns_results["overall_pso"]; og=ns_results["overall_ga"]; ol=ns_results["overall_lstm"]
    print(f"  {'-'*16}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
    print(f"  {'Overall':<16s}  {op:>6.1f}%  {og:>6.1f}%  {ol:>6.1f}%  {max(op,og)-ol:>+.1f}%")

    plot_results(rewards_log,acc_log,results,guide_pso,guide_ga,
                 ns_results,seg_bounds,regime_labels,
                 latency_ms,ppo_train_time,lstm_train_time,CFG)