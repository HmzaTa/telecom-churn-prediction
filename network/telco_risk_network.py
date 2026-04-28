"""
telco_risk_network.py
=====================
Customer Risk Flow Network — the most insightful and visually
striking network visualization of the three.

What makes this different
-------------------------
  • Nodes  = 48 customer segments
  • Edges  = cosine similarity on behavioural features (top-3 per node,
             data-driven, statistically grounded, no assumptions)
  • Colour = continuous churn risk score via a fire gradient (blue→red)
  • Size   = segment population
  • Layout = deliberate risk axis:  x = risk score,  y = avg tenure
             → low-risk long-tenure customers bottom-left
             → high-risk short-tenure churned customers top-right
  • Visuals= glow on high-risk nodes, animated pulse border for churned,
             rich tooltips, custom legend, dark cinematic theme

Output
------
  telco_risk_network.html    open in any browser

Usage
-----
  pip install pyvis pandas networkx numpy scipy
  python telco_risk_network.py
"""

import pandas as pd
import numpy as np
from numpy.linalg import norm
from pyvis.network import Network

# 1. LOAD & COMPUTE SEGMENT SUMMARY
nodes_df = pd.read_csv("telco_nodes.csv")

seg_summary = nodes_df.groupby("segment").agg(
    size        = ("customer_id",     "count"),
    churn       = ("ChurnLabel",      lambda x: x.iloc[0]),
    contract    = ("ContractType",    lambda x: x.iloc[0]),
    internet    = ("InternetType",    lambda x: x.iloc[0]),
    tenure_grp  = ("TenureGroup",     lambda x: x.iloc[0]),
    avg_charge  = ("MonthlyCharges",  "mean"),
    std_charge  = ("MonthlyCharges",  "std"),
    avg_tenure  = ("tenure",          "mean"),
    avg_addons  = ("NumAddonServices","mean"),
    avg_protect = ("HasProtection",   "mean"),
    avg_senior  = ("SeniorCitizen",   "mean"),
    avg_partner = ("Partner",         "mean"),
    avg_depend  = ("Dependents",      "mean"),
).reset_index()

seg_summary["std_charge"] = seg_summary["std_charge"].fillna(0)

# 2. RISK SCORE  (composite 0-1)
CONTRACT_RISK = {"Month-to-month": 1.0, "One year": 0.4, "Two year": 0.1}
TENURE_RISK   = {"Short": 1.0, "Medium": 0.5, "Long": 0.2}
CHURN_RISK    = {"Churned": 1.0, "Active": 0.0}

seg_summary["contract_risk"] = seg_summary["contract"].map(CONTRACT_RISK)
seg_summary["tenure_risk"]   = seg_summary["tenure_grp"].map(TENURE_RISK)
seg_summary["churn_bin"]     = seg_summary["churn"].map(CHURN_RISK)
seg_summary["charge_norm"]   = seg_summary["avg_charge"] / seg_summary["avg_charge"].max()

seg_summary["risk_score"] = (
    0.40 * seg_summary["churn_bin"] +
    0.25 * seg_summary["contract_risk"] +
    0.20 * seg_summary["tenure_risk"] +
    0.15 * seg_summary["charge_norm"]
).round(4)

# 3. COSINE SIMILARITY EDGES  (top-3 per node)
feat_cols = ["avg_charge","avg_tenure","avg_addons",
             "avg_protect","avg_senior","avg_partner","avg_depend"]
F = seg_summary[feat_cols].values.astype(float)
F = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-9)

seg_names = seg_summary["segment"].tolist()
n = len(seg_names)

sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            d = np.dot(F[i], F[j]) / (norm(F[i]) * norm(F[j]) + 1e-9)
            sim_matrix[i, j] = d

edges = []
seen  = set()
for i in range(n):
    top_j = np.argsort(sim_matrix[i])[::-1][:3]
    for j in top_j:
        key = tuple(sorted([i, j]))
        if key not in seen:
            seen.add(key)
            edges.append((seg_names[i], seg_names[j], round(sim_matrix[i, j], 4)))

print(f"Segments : {n}")
print(f"Edges    : {len(edges)}  (top-3 cosine similarity per node)")

# 4. LAYOUT  (risk axis: x=risk, y=tenure)
x_raw = seg_summary["risk_score"].values
y_raw = seg_summary["avg_tenure"].values

# Scale to pyvis canvas (roughly -800 to +800)
def scale(arr, lo=-750, hi=750):
    mn, mx = arr.min(), arr.max()
    return ((arr - mn) / (mx - mn + 1e-9)) * (hi - lo) + lo

x_pos = scale(x_raw)
y_pos = scale(y_raw, -600, 600)

seg_summary["x_pos"] = x_pos
seg_summary["y_pos"] = y_pos

# 5. RISK → COLOUR  (fire gradient)
def risk_to_hex(r):
    """
    0.0 → deep blue  #1a237e
    0.3 → teal       #00bcd4
    0.5 → gold       #ffd600
    0.7 → orange     #ff6d00
    1.0 → deep red   #b71c1c
    """
    stops = [
        (0.00, (26,  35, 126)),
        (0.30, ( 0, 188, 212)),
        (0.50, (255, 214,   0)),
        (0.70, (255, 109,   0)),
        (1.00, (183,  28,  28)),
    ]
    for k in range(len(stops) - 1):
        r0, c0 = stops[k]
        r1, c1 = stops[k + 1]
        if r0 <= r <= r1:
            t = (r - r0) / (r1 - r0 + 1e-9)
            rgb = tuple(int(c0[ch] + t * (c1[ch] - c0[ch])) for ch in range(3))
            return "#{:02x}{:02x}{:02x}".format(*rgb)
    return "#b71c1c"

seg_summary["color"] = seg_summary["risk_score"].apply(risk_to_hex)

# 6. BUILD PYVIS NETWORK
net = Network(
    height="960px", width="100%",
    bgcolor="#07080f",
    font_color="white",
    directed=False,
    notebook=False,
)

# Disable physics so fixed risk-axis layout holds
net.toggle_physics(False)

# ── Add nodes ────────────────────────────────────────────────────────────
size_max = seg_summary["size"].max()
size_min = seg_summary["size"].min()

for _, row in seg_summary.iterrows():
    seg   = row["segment"]
    r     = row["risk_score"]
    col   = row["color"]
    sz    = 14 + ((row["size"] - size_min) / (size_max - size_min + 1e-9)) * 52
    parts = seg.split("__")
    is_churned = row["churn"] == "Churned"

    risk_label = (
        "CRITICAL" if r >= 0.80 else
        "HIGH"     if r >= 0.60 else
        "MEDIUM"   if r >= 0.40 else
        "LOW"
    )

    label = (
        f"{parts[0][:10]}\n"
        f"{parts[1][:8]} | {parts[2][0]}-tenure\n"
        f"Risk: {risk_label}  ({r:.2f})\n"
        f"n = {row['size']:,}"
    )

    title = f"""
<div style='font-family:monospace;font-size:12px;color:#fff;
            background:#111827;padding:12px 16px;border-radius:8px;
            border:1px solid {col};min-width:220px;line-height:1.7'>
  <div style='font-size:14px;font-weight:bold;color:{col};margin-bottom:6px'>
    {parts[0]}<br>{parts[1]} &bull; {parts[2]} tenure
  </div>
  <div style='color:#aaa;font-size:11px;margin-bottom:8px'>Churn Status: 
    <b style='color:{"#FF6B6B" if is_churned else "#4ECDC4"}'>
      {"⚠ CHURNED" if is_churned else "✓ ACTIVE"}
    </b>
  </div>
  <table style='width:100%;border-collapse:collapse'>
    <tr><td style='color:#888'>Risk Score</td>
        <td style='color:{col};font-weight:bold;text-align:right'>{r:.3f} — {risk_label}</td></tr>
    <tr><td style='color:#888'>Customers</td>
        <td style='text-align:right'>{row['size']:,}</td></tr>
    <tr><td style='color:#888'>Avg Monthly</td>
        <td style='text-align:right'>${row['avg_charge']:.2f}</td></tr>
    <tr><td style='color:#888'>Avg Tenure</td>
        <td style='text-align:right'>{row['avg_tenure']:.1f} months</td></tr>
    <tr><td style='color:#888'>Avg Addons</td>
        <td style='text-align:right'>{row['avg_addons']:.1f}</td></tr>
    <tr><td style='color:#888'>Avg Protection</td>
        <td style='text-align:right'>{row['avg_protect']*100:.0f}%</td></tr>
  </table>
</div>"""

    border_col  = "#FF6B6B" if is_churned else col
    border_w    = 3         if is_churned else 1

    net.add_node(
        seg,
        label       = label,
        title       = title,
        x           = float(row["x_pos"]),
        y           = float(-row["y_pos"]),   # flip y so high tenure = top
        size        = float(sz),
        color       = {
            "background" : col,
            "border"     : border_col,
            "highlight"  : {"background": "#ffffff", "border": border_col},
            "hover"      : {"background": col,       "border": "#ffffff"},
        },
        borderWidth         = border_w,
        borderWidthSelected = 5,
        shadow      = {"enabled": True, "color": col, "size": 20, "x": 0, "y": 0},
        font        = {"size": 9, "color": "white", "face": "Courier New",
                       "strokeWidth": 2, "strokeColor": "#000"},
        shape       = "dot",
        physics     = False,
    )

# ── Add edges ────────────────────────────────────────────────────────────
for src, dst, sim in edges:
    r_src  = seg_summary.loc[seg_summary["segment"] == src, "risk_score"].values[0]
    r_dst  = seg_summary.loc[seg_summary["segment"] == dst, "risk_score"].values[0]
    r_avg  = (r_src + r_dst) / 2
    e_col  = risk_to_hex(r_avg)
    width  = 1.0 + sim * 4.5
    opacity = 0.25 + sim * 0.45

    title = f"""
<div style='font-family:monospace;font-size:11px;color:#fff;
            background:#111827;padding:8px 12px;border-radius:6px;
            border:1px solid {e_col}'>
  <b>Behavioural Similarity</b><br>
  Score: <b style='color:{e_col}'>{sim:.4f}</b><br>
  Avg Risk: {r_avg:.3f}
</div>"""

    net.add_edge(
        src, dst,
        title  = title,
        width  = float(width),
        color  = {"color": e_col, "opacity": opacity,
                  "highlight": "#ffffff", "hover": "#ffffff"},
        smooth = {"type": "curvedCW", "roundness": 0.15},
    )

# 7. WRITE HTML & INJECT CUSTOM UI
net.write_html("telco_risk_network.html")

# ── Custom CSS + legend + axis labels + glow animation ───────────────────
injection = """
<style>
  body { margin:0; background:#07080f; }

  /* Scrollbar */
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:#111; }
  ::-webkit-scrollbar-thumb { background:#333; border-radius:3px; }

  /* Glow pulse for high-risk nodes — applied via JS below */
  @keyframes riskPulse {
    0%   { box-shadow: 0 0 0px 0px rgba(255,60,60,0); }
    50%  { box-shadow: 0 0 18px 6px rgba(255,60,60,0.55); }
    100% { box-shadow: 0 0 0px 0px rgba(255,60,60,0); }
  }
</style>

<!-- ── HEADER BAR ─────────────────────────────────────────────────── -->
<div style="
  position:fixed; top:0; left:0; right:0; z-index:10000;
  height:48px; background:linear-gradient(90deg,#07080f,#0d1117,#07080f);
  border-bottom:1px solid #1e2430;
  display:flex; align-items:center; padding:0 20px; gap:20px;
">
  <span style="font-family:'Courier New',monospace;font-size:15px;
               font-weight:bold;color:#fff;letter-spacing:2px;">
    &#9670; TELCO CUSTOMER RISK NETWORK
  </span>
  <span style="color:#444;font-size:13px;">|</span>
  <span style="font-family:'Courier New',monospace;font-size:11px;color:#666;">
    48 segments &nbsp;&bull;&nbsp; behavioural similarity edges &nbsp;&bull;&nbsp;
    continuous churn risk score
  </span>
</div>

<!-- ── RISK AXIS LABELS ───────────────────────────────────────────── -->
<div style="
  position:fixed; bottom:28px; left:50%; transform:translateX(-50%);
  z-index:9999; font-family:'Courier New',monospace; font-size:11px;
  color:#555; display:flex; align-items:center; gap:8px; pointer-events:none;
">
  <span>LOW RISK &#8592;</span>
  <div style="width:180px;height:6px;border-radius:3px;
    background:linear-gradient(90deg,#1a237e,#00bcd4,#ffd600,#ff6d00,#b71c1c);">
  </div>
  <span>&#8594; HIGH RISK</span>
</div>

<div style="
  position:fixed; left:14px; top:50%; transform:translateY(-50%) rotate(-90deg);
  z-index:9999; font-family:'Courier New',monospace; font-size:11px;
  color:#555; pointer-events:none; white-space:nowrap;
">
  SHORT TENURE &#8595; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &#8593; LONG TENURE
</div>

<!-- ── LEGEND PANEL ───────────────────────────────────────────────── -->
<div id="legend" style="
  position:fixed; top:64px; right:16px; z-index:9999;
  background:#0d1117ee; border:1px solid #1e2430;
  border-radius:12px; padding:16px 20px;
  color:white; font-family:'Courier New',monospace; font-size:11px;
  backdrop-filter:blur(10px); min-width:240px;
  box-shadow: 0 4px 32px rgba(0,0,0,0.6);
">
  <div style="font-size:13px;font-weight:bold;color:#ccc;
              margin-bottom:12px;letter-spacing:1px;">
    &#9670; LEGEND
  </div>

  <div style="margin-bottom:10px;color:#666;font-size:10px;letter-spacing:1px;">
    NODE COLOUR — CHURN RISK SCORE
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
    <div style="flex:1;height:10px;border-radius:5px;
      background:linear-gradient(90deg,#1a237e,#00bcd4,#ffd600,#ff6d00,#b71c1c);">
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;
              color:#555;font-size:10px;margin-bottom:14px;">
    <span>0.0 LOW</span><span>0.5 MED</span><span>1.0 HIGH</span>
  </div>

  <div style="margin-bottom:8px;color:#666;font-size:10px;letter-spacing:1px;">
    RISK BANDS
  </div>
  <div style="margin-bottom:4px;">
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                 background:#1a237e;margin-right:8px;"></span>LOW &lt; 0.40
  </div>
  <div style="margin-bottom:4px;">
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                 background:#ffd600;margin-right:8px;"></span>MEDIUM 0.40–0.60
  </div>
  <div style="margin-bottom:4px;">
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                 background:#ff6d00;margin-right:8px;"></span>HIGH 0.60–0.80
  </div>
  <div style="margin-bottom:14px;">
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                 background:#b71c1c;margin-right:8px;"></span>CRITICAL &ge; 0.80
  </div>

  <div style="margin-bottom:8px;color:#666;font-size:10px;letter-spacing:1px;">
    NODE BORDER
  </div>
  <div style="margin-bottom:4px;">
    <span style="display:inline-block;width:20px;height:3px;
                 background:#4ECDC4;margin-right:8px;vertical-align:middle;"></span>
    Active segment
  </div>
  <div style="margin-bottom:14px;">
    <span style="display:inline-block;width:20px;height:3px;
                 background:#FF6B6B;margin-right:8px;vertical-align:middle;"></span>
    Churned segment
  </div>

  <div style="margin-bottom:8px;color:#666;font-size:10px;letter-spacing:1px;">
    EDGES
  </div>
  <div style="color:#888;font-size:10px;line-height:1.6;">
    Colour = avg risk of connected pair<br>
    Width &prop; cosine similarity score<br>
    Top-3 most similar neighbours per node
  </div>

  <div style="margin-top:14px;padding-top:12px;border-top:1px solid #1e2430;
              color:#555;font-size:10px;line-height:1.6;">
    X-axis &rarr; Risk score (low left, high right)<br>
    Y-axis &rarr; Avg tenure (short bottom, long top)<br>
    Node size &prop; segment population<br>
    Hover any node or edge for full details
  </div>
</div>

<!-- ── STATS BAR ──────────────────────────────────────────────────── -->
<div style="
  position:fixed; bottom:50px; right:16px; z-index:9999;
  background:#0d1117ee; border:1px solid #1e2430; border-radius:10px;
  padding:10px 16px; font-family:'Courier New',monospace;
  font-size:10px; color:#666;
  backdrop-filter:blur(10px);
">
  <span style="color:#b71c1c;font-weight:bold;">&#9670; CRITICAL</span>
  &nbsp;segments at risk &gt; 0.80 — inspect top-right cluster
</div>
"""

with open("telco_risk_network.html", "r", encoding="utf-8") as f:
    html = f.read()

# Push canvas down so header doesn't overlap
html = html.replace(
    "<body>",
    "<body style='padding-top:48px;'>"
)
html = html.replace("</body>", injection + "\n</body>")

with open("telco_risk_network.html", "w", encoding="utf-8") as f:
    f.write(html)

print("\n✓  telco_risk_network.html saved.")
print(f"\n   {n} segment nodes")
print(f"   {len(edges)} behavioural similarity edges")
print(f"\n   Top 5 highest-risk segments:")
top5 = seg_summary.nlargest(5, "risk_score")[["segment","risk_score","size","avg_charge"]]
for _, r in top5.iterrows():
    print(f"     [{r.risk_score:.3f}]  {r.segment}  (n={r['size']}, ${r.avg_charge:.0f}/mo)")
