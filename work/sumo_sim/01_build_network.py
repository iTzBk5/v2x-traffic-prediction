

import os, sys, subprocess, urllib.request, urllib.parse, time, math

SUMO_HOME  = r"C:\SUMO"
SUMO_BIN   = os.path.join(SUMO_HOME, "bin")
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")

os.environ["SUMO_HOME"] = SUMO_HOME
os.environ["PATH"] = SUMO_BIN + ";" + os.environ.get("PATH", "")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = os.path.join(SCRIPT_DIR, "network")
os.makedirs(NETWORK_DIR, exist_ok=True)

CENTER_LAT = 44.802850
CENTER_LON = 20.470492
RADIUS_M   = 800                                                  

D_LAT = RADIUS_M / 111320.0
D_LON = RADIUS_M / (111320.0 * abs(math.cos(math.radians(CENTER_LAT))))

SOUTH = CENTER_LAT - D_LAT
NORTH = CENTER_LAT + D_LAT
WEST  = CENTER_LON - D_LON
EAST  = CENTER_LON + D_LON



def download_osm():
    osm_file = os.path.join(NETWORK_DIR, "belgrade.osm.xml")
    if os.path.exists(osm_file):
        sz = os.path.getsize(osm_file)
        if sz > 5_000:
            print(f"  OSM file already exists ({sz/1024:.0f} KB)")
            return osm_file

    bbox_str = f"{WEST},{SOUTH},{EAST},{NORTH}"
    osm_api  = f"https://api.openstreetmap.org/api/0.6/map?bbox={bbox_str}"
    print(f"  Downloading from OSM API")
    try:
        req = urllib.request.Request(osm_api)
        req.add_header("User-Agent", "SUMO-PPO-Sim/1.0")
        resp = urllib.request.urlopen(req, timeout=120)
        raw  = resp.read()
        if len(raw) > 2000:
            with open(osm_file, "wb") as f:
                f.write(raw)
            print(f"  OSM saved, {len(raw)/1024:.0f} KB")
            return osm_file
    except Exception as e:
        print(f"  OSM API failed: {e}")

    bbox_q = f"{SOUTH},{WEST},{NORTH},{EAST}"
    query  = f'[out:xml][timeout:180];(way["highway"]({bbox_q});node(w););out body;>;out skel qt;'
    data   = urllib.parse.urlencode({"data": query}).encode()
    for ep in ["https://overpass.kumi.systems/api/interpreter",
               "https://z.overpass-api.de/api/interpreter",
               "http://overpass-api.de/api/interpreter"]:
        print(f"  Trying {ep}")
        try:
            req = urllib.request.Request(ep, data=data)
            req.add_header("User-Agent", "SUMO-PPO-Sim/1.0")
            resp = urllib.request.urlopen(req, timeout=180)
            raw = resp.read()
            if len(raw) > 2000:
                with open(osm_file, "wb") as f:
                    f.write(raw)
                print(f"  OSM saved, {len(raw)/1024:.0f} KB")
                return osm_file
        except Exception as e:
            print(f"  Failed: {e}")
            time.sleep(2)

    raise RuntimeError("Failed to download OSM data")

def convert_network(osm_file):
    net_file = os.path.join(NETWORK_DIR, "belgrade_tls.net.xml")
    cmd = [
        os.path.join(SUMO_BIN, "netconvert"),
        "--osm-files", osm_file,
        "-o", net_file,
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--output.street-names",
        "--output.original-names",
        "--proj.utm",
    ]
    print("  Converting OSM to SUMO network")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  ERR:", result.stderr[:800])
        raise RuntimeError("netconvert failed")
    print(f"  Network saved: {net_file}")
    return net_file

def generate_demand(net_file):
    rou_file   = os.path.join(NETWORK_DIR, "belgrade_24h.rou.xml")
    trips_file = os.path.join(NETWORK_DIR, "belgrade_24h.trips.xml")
    cmd = [
        sys.executable,
        os.path.join(SUMO_TOOLS, "randomTrips.py"),
        "-n", net_file,
        "-o", trips_file,
        "-r", rou_file,
        "--period", "4.0",                                               
        "-e", "3600",
        "--fringe-factor", "2",
        "--validate",
    ]
    print("  Generating traffic demand")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  Retrying with simpler config")
        cmd2 = [
            sys.executable,
            os.path.join(SUMO_TOOLS, "randomTrips.py"),
            "-n", net_file,
            "-o", trips_file,
            "-r", rou_file,
            "--period", "1.5",
            "-e", "3600",
            "--fringe-factor", "2",
        ]
        result = subprocess.run(cmd2, capture_output=True, text=True)
        if result.returncode != 0:
            print("  ERR:", result.stderr[:500])
            raise RuntimeError("randomTrips.py failed")
    print(f"  Routes saved: {rou_file}")
    return rou_file

def create_sumocfg(net_file, rou_file):
    cfg_file = os.path.join(NETWORK_DIR, "belgrade_24h.sumocfg.xml")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{os.path.basename(net_file)}"/>
        <route-files value="{os.path.basename(rou_file)}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <processing>
        <time-to-teleport value="120"/>
    </processing>
</configuration>"""
    with open(cfg_file, "w") as f:
        f.write(xml)
    print(f"  Config saved: {cfg_file}")
    return cfg_file

if __name__ == "__main__":
    print(f"Building SUMO network for Belgrade ({CENTER_LAT}, {CENTER_LON})")

    print("\nDownloading OSM data")
    osm_file = download_osm()

    print("Converting to SUMO network")
    net_file = convert_network(osm_file)

    print("Generating traffic demand")
    rou_file = generate_demand(net_file)

    print("Creating SUMO config")
    cfg_file = create_sumocfg(net_file, rou_file)

    print(f"\nDone. Run: python 03_compare_simulations.py")
