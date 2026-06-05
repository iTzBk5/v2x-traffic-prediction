import xml.etree.ElementTree as ET
import random, os, math

DURATION_SECONDS = 7200  # 3 hours

def get_edge_centers():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    net_path = os.path.join(script_dir, "network", "belgrade_tls.net.xml")
    tree = ET.parse(net_path)
    root = tree.getroot()
    edge_map = {}
    for e in root.findall("edge"):
        eid = e.get("id")
        if not eid or eid.startswith(":"): continue
        lane = e.find("lane")
        if lane is None: continue
        shape = lane.get("shape", "")
        if not shape: continue
        pts = [(float(p.split(",")[0]), float(p.split(",")[1])) for p in shape.split()]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        edge_map[eid] = (cx, cy)
    return edge_map

def generate():
    print("Reading network")
    edge_map = get_edge_centers()
    edges = list(edge_map.keys())

    edges_by_x = sorted(edges, key=lambda eid: edge_map[eid][0])
    edges_by_y = sorted(edges, key=lambda eid: edge_map[eid][1])

    east_edges  = edges_by_x[int(len(edges)*0.8):]
    west_edges  = edges_by_x[:int(len(edges)*0.2)]
    north_edges = edges_by_y[int(len(edges)*0.8):]
    south_edges = edges_by_y[:int(len(edges)*0.2)]

    print(f"Generating 3h traffic ({DURATION_SECONDS}s)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "network", "belgrade_3h.rou.xml")

    with open(out_path, "w") as f:
        f.write("<routes>\n")
        f.write('  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" '
                'length="5" minGap="2" maxSpeed="15.0" guiShape="passenger"/>\n')

        vid = 0
        for step in range(DURATION_SECONDS):
            # Shift the hour so step=0 is 07:00 AM
            hour = 7.0 + (step / 3600.0)
            
            # Massive morning & evening rushes
            morning_rush = math.exp(-((hour - 8.0)**2) / 3.0)
            evening_rush = math.exp(-((hour - 17.0)**2) / 3.0)
            night_lull   = math.exp(-((hour - 3.0)**2) / 8.0)
            
            # Flow factor ranges from ~0.3 (night) to 1.5 (moderate rush)
            flow_factor = 0.3 + 1.2 * (morning_rush + evening_rush) - 0.15 * night_lull
            flow_factor = max(0.15, min(flow_factor, 1.5))

            # Random background traffic everywhere
            # At peak (flow_factor=1.5), interval is 2 -> 0.5 cars/second
            bg_interval = max(1, int(2.0 / flow_factor))
            if step % bg_interval == 0:
                e1, e2 = random.sample(edges, 2)
                f.write(f'  <trip id="bg_{vid}" type="car" depart="{step}" from="{e1}" to="{e2}"/>\n')
                vid += 1

            # Directional traffic (East-West / North-South)
            ew_interval = max(1, int(2.0 / flow_factor))
            if step % ew_interval == 0 and random.random() < 0.65:
                if hour < 12:
                    e1 = random.choice(east_edges)
                    e2 = random.choice(west_edges)
                else:
                    e1 = random.choice(west_edges)
                    e2 = random.choice(east_edges)
                f.write(f'  <trip id="ew_{vid}" type="car" depart="{step}" from="{e1}" to="{e2}"/>\n')
                vid += 1

            ns_interval = max(1, int(2.0 / flow_factor))
            if step % ns_interval == 0 and random.random() < 0.65:
                if morning_rush > evening_rush:
                    e1 = random.choice(north_edges)
                    e2 = random.choice(south_edges)
                else:
                    e1 = random.choice(south_edges)
                    e2 = random.choice(north_edges)
                f.write(f'  <trip id="ns_{vid}" type="car" depart="{step}" from="{e1}" to="{e2}"/>\n')
                vid += 1

            if step > 0 and step % 3600 == 0:
                print(f"  Hour {int(hour)} done ({vid:,} trips generated)")

        f.write("</routes>\n")

    print(f"Generated {vid:,} trips")
    print(f"Output: {out_path}")

    # Create SUMOCFG
    cfg_path = os.path.join(script_dir, "network", "belgrade_3h.sumocfg.xml")
    with open(cfg_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<configuration>\n')
        f.write('    <input>\n')
        f.write('        <net-file value="belgrade_tls.net.xml"/>\n')
        f.write('        <route-files value="belgrade_3h.rou.xml"/>\n')
        f.write('    </input>\n')
        f.write('    <time>\n')
        f.write(f'        <begin value="0"/>\n')
        f.write(f'        <end value="{DURATION_SECONDS}"/>\n')
        f.write('    </time>\n')
        f.write('    <processing>\n')
        f.write('        <time-to-teleport value="300"/>\n')
        f.write('    </processing>\n')
        f.write('</configuration>\n')
    print(f"Config saved: {cfg_path}")

if __name__ == "__main__":
    generate()
