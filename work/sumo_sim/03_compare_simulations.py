

import os, sys, math, time, csv, json
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

SUMO_HOME = r"C:\SUMO"
os.environ["SUMO_HOME"] = SUMO_HOME
os.environ["PATH"] = os.path.join(SUMO_HOME, "bin") + ";" + os.environ.get("PATH", "")
sys.path.insert(0, os.path.join(SUMO_HOME, "tools"))

import traci

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from model_def import (DirectionActorCritic, OnlineMetaheuristicGuide,
                        CFG, DIR_NAMES, DIR_ARROWS)

WORK_DIR      = os.path.dirname(SCRIPT_DIR)
MODEL_PATH    = os.path.join(WORK_DIR,  "ppo_best.pt")
NETWORK_DIR   = os.path.join(SCRIPT_DIR, "network")
NET_XML       = os.path.join(NETWORK_DIR, "belgrade_tls.net.xml")
SUMOCFG_24H   = os.path.join(NETWORK_DIR, "belgrade_3h.sumocfg.xml")
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFER_INTERVAL   = 30     # react faster
WARMUP_SECS      = 120    # short warmup — let adaptive control act early
MIN_CONFIDENCE   = 0.40   # lower threshold for prediction display
MAX_PHASE_ADJUST = 8      # not used by max-pressure, kept for reference
MP_MIN_GREEN     = 10     # minimum green time before max-pressure can switch (seconds)
MP_DECISION_SEC  = 5      # how often max-pressure re-evaluates (seconds)
DIR_BOOST_FACTOR = 1.5    # how strongly the predicted direction boosts phase pressure
DIR_BOOST_MIN_CONF = 0.55 # only apply boost when model confidence exceeds this threshold
CMAP = mpl_cm.get_cmap("hot")


# Running normalizer

class RunningNormalizer:
    def __init__(self, shape, momentum=0.01):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var  = np.ones(shape,  dtype=np.float32)
        self.count = 0
        self.mom   = momentum

    def update(self, x):
        self.count += 1
        if self.count == 1:
            self.mean = x.copy().astype(np.float32)
            self.var  = np.ones_like(self.mean)
        else:
            self.mean = (1 - self.mom) * self.mean + self.mom * x
            self.var  = (1 - self.mom) * self.var  + self.mom * (x - self.mean)**2

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-6)


# Geometry helpers

def xy_bearing(x1, y1, x2, y2):
    return math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360

def bearing_to_dir(b):
    return int((b + 45) % 360 // 90)

def heat_to_sumo_color(heat, alpha=200):
    r, g, b, _ = CMAP(float(np.clip(heat, 0, 1)))
    return (int(r * 255), int(g * 255), int(b * 255), alpha)

def make_circle(cx, cy, radius, n_pts=48):
    pts = []
    for i in range(n_pts):
        a = 2 * math.pi * i / n_pts
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    pts.append(pts[0])
    return pts

def offset_shape(pts, offset_m=2.0):
    left, right = [], []
    n = len(pts)
    for i in range(n):
        if i == 0:
            dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        elif i == n - 1:
            dx, dy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
        else:
            dx, dy = pts[i+1][0] - pts[i-1][0], pts[i+1][1] - pts[i-1][1]
        length = max(math.hypot(dx, dy), 0.01)
        px, py = dy / length, -dx / length
        x, y = pts[i]
        left.append((x - px * offset_m, y - py * offset_m))
        right.append((x + px * offset_m, y + py * offset_m))
    return left + list(reversed(right))


# Network parser

def parse_network():
    tree = ET.parse(NET_XML)
    root = tree.getroot()

    edge_shapes, edge_dir_map = {}, {}
    for e in root.findall("edge"):
        eid = e.get("id", "")
        if eid.startswith(":"):
            continue
        lane = e.find("lane")
        if lane is None:
            continue
        shape_str = lane.get("shape", "")
        if not shape_str:
            continue
        try:
            pts = [(float(p.split(",")[0]), float(p.split(",")[1]))
                   for p in shape_str.strip().split()]
            if len(pts) < 2:
                continue
        except:
            continue
        edge_shapes[eid] = pts
        x1, y1 = pts[0]; x2, y2 = pts[-1]
        d = bearing_to_dir(xy_bearing(x1, y1, x2, y2)) if math.hypot(x2-x1, y2-y1) > 0.5 else 0
        edge_dir_map[eid] = d

    tls_link_dirs = {}
    for c in root.findall("connection"):
        tl = c.get("tl")
        if not tl:
            continue
        li = int(c.get("linkIndex", -1))
        fe = c.get("from", "")
        if fe in edge_shapes:
            pts = edge_shapes[fe]
            d = bearing_to_dir(xy_bearing(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]))
            tls_link_dirs.setdefault(tl, {})[li] = d

    tls_phases, tls_phase_dirs = {}, {}
    for tl_elem in root.findall("tlLogic"):
        tl_id = tl_elem.get("id")
        phases_info, dir_sets = [], []
        for ph in tl_elem.findall("phase"):
            dur = int(ph.get("duration"))
            state = ph.get("state")
            phases_info.append((dur, state))
            gd = set()
            ld = tls_link_dirs.get(tl_id, {})
            for i, c in enumerate(state):
                if c in "Gg" and i in ld:
                    gd.add(ld[i])
            dir_sets.append(gd)
        tls_phases[tl_id] = phases_info
        tls_phase_dirs[tl_id] = dir_sets

    junction_coords = {}
    all_junction_coords = {}
    for j in root.findall("junction"):
        jid = j.get("id")
        jtype = j.get("type", "")
        if jid.startswith(":"):
            continue
        try:
            x = float(j.get("x"))
            y = float(j.get("y"))
        except:
            continue
        if jtype == "traffic_light":
            junction_coords[jid] = (x, y)
        if jtype in ("traffic_light", "priority", "right_before_left"):
            all_junction_coords[jid] = (x, y)

    print(f"Network: {len(edge_shapes)} edges, {len(tls_phases)} TLS, {len(junction_coords)} signalized, {len(all_junction_coords)} total junctions")
    return edge_shapes, edge_dir_map, tls_phases, tls_phase_dirs, junction_coords, all_junction_coords


def compute_direction_density(conn, rsu_xy, n_rsu, n_dir, v2r_range):
    D = np.zeros((n_rsu, n_dir), dtype=np.float32)
    vehs = conn.vehicle.getIDList()
    for vid in vehs:
        try:
            vx, vy = conn.vehicle.getPosition(vid)
            angle  = conn.vehicle.getAngle(vid)
            veh_dir = bearing_to_dir(angle)
            for i, (rx, ry) in enumerate(rsu_xy):
                if math.hypot(vx - rx, vy - ry) <= v2r_range:
                    D[i, veh_dir] += 1.0
        except:
            continue
    return D


class StatsCollector:
    def __init__(self):
        self.step_data = []
        self.all_speeds = []
        self.all_waiting = []
        self.total_departed = 0
        self.total_arrived = 0
        self.total_teleported = 0

    def collect(self, conn, sim_time):
        vehs = conn.vehicle.getIDList()
        speeds, waiting, halted = [], [], 0
        for vid in vehs:
            try:
                s = conn.vehicle.getSpeed(vid) * 3.6
                w = conn.vehicle.getWaitingTime(vid)
                speeds.append(s)
                waiting.append(w)
                self.all_speeds.append(s)
                self.all_waiting.append(w)
                if s < 0.5:
                    halted += 1
            except:
                pass
        self.total_departed  += len(conn.simulation.getDepartedIDList())
        self.total_arrived   += len(conn.simulation.getArrivedIDList())
        self.total_teleported += conn.simulation.getStartingTeleportNumber()

        self.step_data.append({
            "sim_time":      sim_time,
            "n_vehicles":    len(vehs),
            "avg_speed_kmh": round(float(np.mean(speeds)) if speeds else 0, 2),
            "avg_waiting_s": round(float(np.mean(waiting)) if waiting else 0, 2),
            "max_waiting_s": round(float(np.max(waiting)) if waiting else 0, 2),
            "halted":        halted,
        })

    def summary(self):
        if not self.step_data:
            return {}
        return {
            "avg_speed_kmh":    round(float(np.mean(self.all_speeds)) if self.all_speeds else 0, 2),
            "avg_waiting_s":    round(float(np.mean(self.all_waiting)) if self.all_waiting else 0, 2),
            "median_waiting_s": round(float(np.median(self.all_waiting)) if self.all_waiting else 0, 2),
            "p95_waiting_s":    round(float(np.percentile(self.all_waiting, 95)) if self.all_waiting else 0, 2),
            "avg_halted_vehs":  round(float(np.mean([d["halted"] for d in self.step_data])), 1),
            "avg_active_vehs":  round(float(np.mean([d["n_vehicles"] for d in self.step_data])), 1),
            "peak_vehicles":    max(d["n_vehicles"] for d in self.step_data),
            "total_departed":   self.total_departed,
            "throughput":       self.total_arrived,
            "total_teleported": self.total_teleported,
        }


def generate_charts(b_sum, a_sum, b_steps, a_steps):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0D1117")
    fig.suptitle("MEC Adaptive Control vs Fixed-Time Baseline",
                 fontsize=16, color="#E6EDF3", fontweight="bold", y=0.98)
    for ax in axes.flat:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#8B949E", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#30363D")
        ax.grid(True, alpha=0.15, color="#30363D")

    t_b = [d["sim_time"]/3600 for d in b_steps]
    t_a = [d["sim_time"]/3600 for d in a_steps]

    ax = axes[0,0]
    ax.plot(t_b, [d["avg_speed_kmh"] for d in b_steps], color="#FF6B6B", label="Fixed-Time", lw=1.5)
    ax.plot(t_a, [d["avg_speed_kmh"] for d in a_steps], color="#2ED573", label="MEC Adaptive", lw=1.5)
    ax.set_title("Average Speed (km/h)", color="#E6EDF3", fontsize=11)
    ax.set_xlabel("Time (Hours)", color="#8B949E")
    ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#E6EDF3")

    ax = axes[0,1]
    ax.plot(t_b, [d["avg_waiting_s"] for d in b_steps], color="#FF6B6B", label="Fixed-Time", lw=1.5)
    ax.plot(t_a, [d["avg_waiting_s"] for d in a_steps], color="#2ED573", label="MEC Adaptive", lw=1.5)
    ax.set_title("Average Waiting Time (s)", color="#E6EDF3", fontsize=11)
    ax.set_xlabel("Time (Hours)", color="#8B949E")
    ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#E6EDF3")

    ax = axes[0,2]
    ax.plot(t_b, [d["n_vehicles"] for d in b_steps], color="#FF6B6B", label="Fixed-Time", lw=1.5)
    ax.plot(t_a, [d["n_vehicles"] for d in a_steps], color="#2ED573", label="MEC Adaptive", lw=1.5)
    ax.set_title("Active Vehicles", color="#E6EDF3", fontsize=11)
    ax.set_xlabel("Time (Hours)", color="#8B949E")
    ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#E6EDF3")

    ax = axes[1,0]
    ax.plot(t_b, [d["halted"] for d in b_steps], color="#FF6B6B", label="Fixed-Time", lw=1.5)
    ax.plot(t_a, [d["halted"] for d in a_steps], color="#2ED573", label="MEC Adaptive", lw=1.5)
    ax.set_title("Halted Vehicles", color="#E6EDF3", fontsize=11)
    ax.set_xlabel("Time (Hours)", color="#8B949E")
    ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#E6EDF3")

    ax = axes[1,1]
    metrics = ["avg_speed_kmh", "avg_waiting_s", "throughput", "avg_halted_vehs"]
    labels  = ["Avg Speed\n(km/h)", "Avg Wait\n(s)", "Throughput", "Avg Halted"]
    bv = [b_sum.get(m,0) for m in metrics]
    av = [a_sum.get(m,0) for m in metrics]
    x = np.arange(len(metrics)); w = 0.35
    bb = ax.bar(x-w/2, bv, w, label="Fixed-Time", color="#FF6B6B", alpha=0.8, edgecolor="#FF4444")
    ba = ax.bar(x+w/2, av, w, label="MEC Adaptive", color="#2ED573", alpha=0.8, edgecolor="#27AE60")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, color="#C9D1D9")
    ax.set_title("Key Metrics", color="#E6EDF3", fontsize=11)
    ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#E6EDF3")
    for bar in list(bb)+list(ba):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=7, color="#C9D1D9")

    ax = axes[1,2]; ax.axis("off")
    y = 0.92
    ax.text(0.5, y, "IMPROVEMENT SUMMARY", fontsize=13, color="#58A6FF",
            fontweight="bold", ha="center", va="top", family="monospace",
            transform=ax.transAxes)
    y -= 0.10
    def pct(bval, aval, higher_is_better=True):
        if abs(bval) < 0.01: return 0
        c = ((aval - bval) / abs(bval)) * 100
        return c if higher_is_better else -c

    items = [
        ("Avg Speed",    pct(b_sum.get("avg_speed_kmh",1), a_sum.get("avg_speed_kmh",1), True)),
        ("Waiting Time", pct(b_sum.get("avg_waiting_s",1), a_sum.get("avg_waiting_s",1), False)),
        ("Throughput",   pct(b_sum.get("throughput",1), a_sum.get("throughput",1), True)),
        ("Halted Vehs",  pct(b_sum.get("avg_halted_vehs",1), a_sum.get("avg_halted_vehs",1), False)),
    ]
    for label, imp in items:
        color = "#2ED573" if imp > 0 else "#FF6B6B"
        arrow = "^" if imp > 0 else "v"
        ax.text(0.12, y, f"{label}:", fontsize=11, color="#C9D1D9",
                va="top", family="monospace", transform=ax.transAxes)
        ax.text(0.72, y, f"{arrow} {imp:+.1f}%", fontsize=12, color=color,
                fontweight="bold", va="top", family="monospace",
                transform=ax.transAxes)
        y -= 0.10

    plt.tight_layout(rect=[0,0,1,0.95])
    path = os.path.join(OUTPUT_DIR, "comparison_chart.png")
    fig.savefig(path, dpi=150, facecolor="#0D1117", bbox_inches="tight")
    print(f"Chart saved: {path}")
    plt.show()


def main():
    print("Starting comparison")

    print("Parsing network")
    edge_shapes, edge_dir_map, tls_phases, tls_phase_dirs, junction_coords, all_junction_coords = parse_network()

    edge_ribbons = {}
    for eid, pts in edge_shapes.items():
        edge_ribbons[eid] = offset_shape(pts, 2.0)

    print("Loading model")
    model = DirectionActorCritic(CFG).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded, {n_params:,} params")

    guide = OnlineMetaheuristicGuide(
        n_dir=CFG["n_dir"], method="GA", cfg=CFG)
    print("GA online guide ready")

    print("Starting SUMO instances")
    sumo_bin = os.path.join(SUMO_HOME, "bin", "sumo-gui")

    traci.start([sumo_bin, "-c", SUMOCFG_24H,
                 "--start", "--delay", "0", "--quit-on-end",
                 "--window-pos", "0,0", "--window-size", "960,1080",
                 "--ignore-route-errors", "true"],
                label="baseline")
    conn_b = traci.getConnection("baseline")
    print("Baseline started")

    traci.start([sumo_bin, "-c", SUMOCFG_24H,
                 "--start", "--delay", "0", "--quit-on-end",
                 "--window-pos", "960,0", "--window-size", "960,1080",
                 "--ignore-route-errors", "true"],
                label="ai_control")
    conn_a = traci.getConnection("ai_control")
    print("AI control started")

    print("Placing RSUs within gNB coverage")
    n_rsu_target = CFG["n_rsu"]
    gnb_range    = CFG["gnb_range"]

    candidate_pool = {**all_junction_coords}
    if len(candidate_pool) == 0:
        raise ValueError("No junctions found!")

    all_jx = [p[0] for p in candidate_pool.values()]
    all_jy = [p[1] for p in candidate_pool.values()]
    gnb_cx = np.mean(all_jx)
    gnb_cy = np.mean(all_jy)
    gnb_center_xy = (gnb_cx, gnb_cy)

    in_range = {}
    for jid, (jx, jy) in candidate_pool.items():
        dist = math.hypot(jx - gnb_cx, jy - gnb_cy)
        if dist <= gnb_range:
            in_range[jid] = (jx, jy, dist)

    def junction_priority(jid):
        is_tls = jid in junction_coords
        n_phases = len(tls_phases.get(jid, []))
        dist = in_range[jid][2]
        return (-int(is_tls), -n_phases, dist)

    sorted_jids = sorted(in_range.keys(), key=junction_priority)

    sel_ids, sel_coords = [], []
    for min_dist in [300, 200, 150, 100, 50, 0]:
        sel_ids, sel_coords = [], []
        for jid in sorted_jids:
            if len(sel_ids) >= n_rsu_target:
                break
            jx, jy = in_range[jid][0], in_range[jid][1]
            if min_dist > 0 and not all(
                    math.hypot(jx - sx, jy - sy) >= min_dist
                    for sx, sy in sel_coords):
                continue
            sel_ids.append(jid)
            sel_coords.append((jx, jy))
        if len(sel_ids) >= n_rsu_target:
            print(f"Selected {len(sel_ids)} RSUs")
            break
    else:
        print(f"Only found {len(sel_ids)} junctions")

    rsu_xy_a = sel_coords
    rsu_xy_b = list(sel_coords)

    gnb_xy_b = gnb_center_xy
    gnb_xy_a = gnb_center_xy

    print("Drawing overlays on AI window")
    _base_colors = [
        (255,107,107), (255,179,64), (46,213,115), (156,136,255),
        (255,71,87),   (30,144,255), (255,215,0),  (0,206,209),
        (255,105,180), (50,205,50),  (255,165,0),  (138,43,226),
        (0,191,255),   (255,99,71),  (60,179,113), (218,112,214),
        (127,255,0),   (255,140,0),  (70,130,180), (186,85,211),
    ]
    n_rsu = len(rsu_xy_a)
    RSU_COLORS = (_base_colors * ((n_rsu // len(_base_colors)) + 1))[:n_rsu]
    gx, gy = gnb_xy_a
    conn_a.polygon.add("gnb_range", make_circle(gx, gy, gnb_range),
                       color=(30,144,255,35), fill=True, layer=1)
    conn_a.poi.add("gnb_tower", gx, gy, color=(30,144,255,220),
                   layer=12, width=30, height=30)
    for i, (x, y) in enumerate(rsu_xy_a):
        r, g, b = RSU_COLORS[i]
        conn_a.polygon.add(f"rsu{i}_range", make_circle(x, y, CFG["v2r_range"]),
                           color=(r,g,b,45), fill=True, layer=2)
        conn_a.poi.add(f"rsu{i}", x, y, color=(r,g,b,230),
                       layer=13, width=16, height=16)

    print(f"    {n_rsu} RSU overlays drawn")

    for conn, rsu_xy in [(conn_b, rsu_xy_b), (conn_a, rsu_xy_a)]:
        cx = np.mean([p[0] for p in rsu_xy])
        cy = np.mean([p[1] for p in rsu_xy])
        conn.gui.setZoom("View #0", 1200)
        conn.gui.setOffset("View #0", cx, cy)

    # ── 7. Simulation state ──────────────────────────────────────────
    n_rsu   = len(rsu_xy_a)
    n_dir   = CFG["n_dir"]
    seq_len = CFG["seq_len"]
    sd      = CFG["state_dim"]

    history_X = deque(maxlen=seq_len)
    history_D = deque(maxlen=seq_len)

    norm_X = RunningNormalizer((n_rsu, sd))
    norm_D = RunningNormalizer((n_rsu, n_dir))

    dir_probs = np.ones(4) / 4
    pred_dir  = 0
    conf      = 0.25
    last_infer_time = -999

    stats_b = StatsCollector()
    stats_a = StatsCollector()

    # ── 8. TLS phase → direction mapping ─────────────────────────────
    tls_favored_phase = {}
    for tl_id, phase_dirs in tls_phase_dirs.items():
        orig = tls_phases.get(tl_id, [])
        ns_phases = []
        ew_phases = []
        for pi, (dur, st) in enumerate(orig):
            is_yellow = all(c in "yr" for c in st)
            if is_yellow:
                continue
            dirs = phase_dirs[pi]
            if dirs & {0, 2}:
                ns_phases.append(pi)
            if dirs & {1, 3}:
                ew_phases.append(pi)
        best_ns = max(ns_phases, key=lambda p: orig[p][0]) if ns_phases else 0
        best_ew = max(ew_phases, key=lambda p: orig[p][0]) if ew_phases else 0
        tls_favored_phase[tl_id] = {"ns": best_ns, "ew": best_ew}

    print("\nRunning simulation")

    max_steps = 10800
    stats_every = 60
    last_tls_phase = {}
    sim_start_time = None

    # ── Match-funded extensions replaced by MAX-PRESSURE control ─────
    # Track when each TLS last switched phase + current phase index
    mp_last_switch  = {}   # tl_id -> sim_time when phase was last set
    mp_current_phase = {}  # tl_id -> phase index we chose
    mp_switch_count = 0    # total phase switches for dashboard

    try:
        for step in range(max_steps):
            conn_b.simulationStep()
            conn_a.simulationStep()

            sim_t   = conn_a.simulation.getTime()
            if sim_start_time is None:
                sim_start_time = sim_t
            n_veh_b = conn_b.vehicle.getIDCount()
            n_veh_a = conn_a.vehicle.getIDCount()

            # ── Data Collection & Inference every INFER_INTERVAL seconds ──
            if sim_t - last_infer_time >= INFER_INTERVAL or step == 0:
                # ── Collect state features X[n_rsu, state_dim] ───────────
                feats = np.zeros((n_rsu, sd), dtype=np.float32)
                vehs = conn_a.vehicle.getIDList()
                rsu_vehs = {i: [] for i in range(n_rsu)}
                for vid in vehs:
                    try:
                        vx, vy = conn_a.vehicle.getPosition(vid)
                        for i, (rx, ry) in enumerate(rsu_xy_a):
                            if math.hypot(vx-rx, vy-ry) <= CFG["v2r_range"]:
                                rsu_vehs[i].append(vid)
                    except:
                        continue
                hour = (sim_t / 3600.0) % 24.0
                for i in range(n_rsu):
                    vl = rsu_vehs[i]; n = len(vl)
                    density = min(n * 2.5, 80.0)
                    if vl:
                        spd=[]; halt=0; waits=[]
                        for vid in vl:
                            try:
                                s = conn_a.vehicle.getSpeed(vid)*3.6
                                spd.append(s)
                                waits.append(conn_a.vehicle.getWaitingTime(vid))
                                if s < 0.5: halt += 1
                            except: pass
                        speed = np.clip(np.mean(spd) if spd else 30, 5, 60)
                        w_avg = np.mean(waits) if waits else 0
                    else:
                        speed, halt, w_avg = 50.0, 0, 0.0
                    queue = min(halt*2.0, 30.0)
                    delay = np.clip(w_avg*0.3, 0, 20)
                    feats[i] = [density, speed, queue, delay,
                                np.clip(0.01+density*0.001, 0, 0.15),
                                np.clip(1+density*0.02, 1, 10),
                                np.clip(density/80, 0, 1),
                                math.sin(2*math.pi*hour/24)]

                # ── Collect direction density D[n_rsu, n_dir] ────────────
                dir_dens = compute_direction_density(
                    conn_a, rsu_xy_a, n_rsu, n_dir, CFG["v2r_range"])

                # ── Normalize and buffer ─────────────────────────────────
                norm_X.update(feats)
                norm_D.update(dir_dens)
                feats_n   = norm_X.normalize(feats)
                dir_dens_n = norm_D.normalize(dir_dens)

                history_X.append(feats_n)
                history_D.append(dir_dens_n)

                # ── Model Inference ──────────────────────────────────────
                buf_X = list(history_X)
                buf_D = list(history_D)
                nh = len(buf_X)
                if nh < seq_len:
                    # Pad with the earliest available data to avoid zero-shock
                    first_X = buf_X[0] if buf_X else np.zeros((n_rsu, sd), dtype=np.float32)
                    first_D = buf_D[0] if buf_D else np.zeros((n_rsu, n_dir), dtype=np.float32)
                    pad_X = [first_X] * (seq_len - nh)
                    pad_D = [first_D] * (seq_len - nh)
                    win_X = np.array(pad_X + buf_X)
                    win_D = np.array(pad_D + buf_D)
                else:
                    win_X = np.array(buf_X[-seq_len:])
                    win_D = np.array(buf_D[-seq_len:])

                obs = np.concatenate([win_X, win_D], axis=-1).flatten().astype(np.float32)
                xt = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                dir_probs, _, raw_logits = model.predict_direction(xt)

                guided_dir = guide.predict(raw_logits)
                pred_dir = guided_dir

                # Ground truth for PSO observation (current peak direction)
                true_label = np.argmax(np.mean(dir_dens, axis=0))
                guide.observe(raw_logits, true_label)

                last_infer_time = sim_t

                # Dashboard print
                trend_str = DIR_NAMES[pred_dir]
                arrow = DIR_ARROWS[pred_dir]
                conf = float(dir_probs[pred_dir])
                print(f"Time {int(sim_t/60):2d}m | MEC Predict: {trend_str} {arrow} | Conf: {conf:.1%} | Switches: {mp_switch_count} | BiasAvg: {np.mean(np.abs(guide.bias)):.3f}")

                # (Heatmap update removed for performance)

                # ── Dashboard ────────────────────────────────────────
                bias_str = [f"{b:+.2f}" for b in guide.bias]
                t_m, t_s = divmod(int(sim_t), 60)
                print(f"  {t_m:02d}:{t_s:02d}  "
                      f"BASELINE vehs={n_veh_b:3d}  |  "
                      f"AI vehs={n_veh_a:3d}  "
                      f"pred={DIR_ARROWS[pred_dir]} {DIR_NAMES[pred_dir]} "
                      f"({dir_probs[pred_dir]*100:.0f}%)  "
                      f"bias={bias_str}")

            # ── TLS control: PER-INTERSECTION MAX-PRESSURE + DRL BOOST ────────
            #    For each signalized intersection, compute queue pressure for
            #    each green phase. Switch to the phase with highest pressure.
            #    INTEGRATION: The DRL direction prediction actively boosts the
            #    pressure of phases serving the predicted dominant direction,
            #    making the system proactive (anticipatory) rather than purely
            #    reactive. Boost = DIR_BOOST_FACTOR × model confidence.
            elapsed = sim_t - sim_start_time
            if elapsed >= WARMUP_SECS and sim_t % MP_DECISION_SEC < 1.0:
                for tl_id, phase_list in tls_phases.items():
                    if len(phase_list) < 2:
                        continue

                    # Enforce minimum green time
                    last_sw = mp_last_switch.get(tl_id, 0)
                    if sim_t - last_sw < MP_MIN_GREEN:
                        continue

                    # Get controlled lanes for this TLS
                    try:
                        ctrl_lanes = conn_a.trafficlight.getControlledLanes(tl_id)
                    except:
                        continue

                    # Compute pressure for each GREEN phase
                    phase_pressure = []
                    for pi, (dur, state) in enumerate(phase_list):
                        # Skip yellow/all-red phases
                        if all(c in "yr" for c in state):
                            phase_pressure.append(-999)
                            continue

                        # Sum halted vehicles on lanes that get green in this phase
                        pressure = 0.0
                        for li, ch in enumerate(state):
                            if ch in "Gg" and li < len(ctrl_lanes):
                                try:
                                    lane = ctrl_lanes[li]
                                    halted = conn_a.lane.getLastStepHaltingNumber(lane)
                                    waiting = conn_a.lane.getWaitingTime(lane)
                                    # Pressure = halted count + waiting time factor
                                    pressure += halted + waiting * 0.1
                                except:
                                    pass

                        # ── DRL Direction Boost ──────────────────────────
                        # If this phase serves the predicted dominant direction
                        # AND the model is confident enough, add a scaled bonus.
                        # Uses conf² so low-confidence predictions barely affect
                        # the max-pressure decision, while high-confidence ones
                        # provide meaningful proactive prioritization.
                        phase_dir_sets = tls_phase_dirs.get(tl_id, [])
                        if (conf >= DIR_BOOST_MIN_CONF
                                and pi < len(phase_dir_sets)
                                and pred_dir in phase_dir_sets[pi]):
                            direction_boost = DIR_BOOST_FACTOR * (conf ** 2)
                            pressure += direction_boost

                        phase_pressure.append(pressure)

                    if not phase_pressure or max(phase_pressure) <= 0:
                        continue

                    # Select phase with maximum pressure
                    best_phase = int(np.argmax(phase_pressure))
                    cur_phase = mp_current_phase.get(tl_id, -1)

                    # Only switch if a different phase has significantly more pressure
                    if best_phase != cur_phase and phase_pressure[best_phase] > 0:
                        cur_pressure = phase_pressure[cur_phase] if 0 <= cur_phase < len(phase_pressure) else 0
                        # Switch if best phase has >20% more pressure than current
                        if cur_pressure <= 0 or phase_pressure[best_phase] > cur_pressure * 1.2:
                            try:
                                conn_a.trafficlight.setPhase(tl_id, best_phase)
                                mp_current_phase[tl_id] = best_phase
                                mp_last_switch[tl_id] = sim_t
                                mp_switch_count += 1
                            except:
                                pass

            # ── Observe true direction for PSO learning ──────────────
            if sim_t - last_infer_time < 1.0 and step > 0:
                actual_D = compute_direction_density(
                    conn_a, rsu_xy_a, n_rsu, n_dir, CFG["v2r_range"])
                dominant_dir = int(np.argmax(actual_D.sum(axis=0)))
                guide.observe(raw_logits, dominant_dir)

            # ── Stats collection ─────────────────────────────────────
            if step % stats_every == 0:
                stats_b.collect(conn_b, sim_t)
                stats_a.collect(conn_a, sim_t)

            if (conn_b.simulation.getMinExpectedNumber() <= 0 and
                conn_a.simulation.getMinExpectedNumber() <= 0):
                print("\n  Both simulations ended.")
                break

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    finally:
        for conn, label in [(conn_b, "baseline"), (conn_a, "ai_control")]:
            try:
                fp = os.path.join(OUTPUT_DIR, f"final_{label}.png")
                conn.gui.screenshot("View #0", fp, width=1920, height=1080)
                print(f"  Screenshot: {fp}")
            except: pass
        conn_b.close()
        conn_a.close()

    # ── Results ──────────────────────────────────────────────────────
    b_sum = stats_b.summary()
    a_sum = stats_a.summary()

    print("\nFinal comparison:")
    print(f"  {'Metric':25s}  {'Fixed-Time':>12s}  {'MEC-Adapt':>12s}  {'Change':>10s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}  {'-'*10}")
    for k in b_sum:
        bv = b_sum[k]; av = a_sum.get(k, 0)
        if isinstance(bv, (int, float)) and abs(bv) > 0.01:
            ch = ((av - bv) / abs(bv)) * 100
            print(f"  {k:25s}  {bv:12.2f}  {av:12.2f}  {ch:+9.1f}%")
        else:
            print(f"  {k:25s}  {str(bv):>12s}  {str(av):>12s}")

    # ── GA guide stats ──────────────────────────────────────────────
    if guide.running_correct:
        ga_acc = np.mean(guide.running_correct) * 100
        print(f"\n  GA online guidance accuracy: {ga_acc:.1f}%")
        print(f"  GA final bias: {[f'{b:+.3f}' for b in guide.bias]}")
        print(f"  GA updates: {len(guide.bias_snapshots)}")

    for label, s in [("baseline", b_sum), ("ai_control", a_sum)]:
        with open(os.path.join(OUTPUT_DIR, f"{label}_stats.json"), "w") as f:
            json.dump(s, f, indent=2)

    for label, steps in [("baseline", stats_b.step_data), ("ai_control", stats_a.step_data)]:
        if steps:
            cp = os.path.join(OUTPUT_DIR, f"{label}_timesteps.csv")
            with open(cp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=steps[0].keys())
                w.writeheader()
                w.writerows(steps)

    print("\n  Generating comparison charts")
    generate_charts(b_sum, a_sum, stats_b.step_data, stats_a.step_data)

    print("\n  Done! Output in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
