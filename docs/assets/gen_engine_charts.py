#!/usr/bin/env python3
"""Generate the 2026-11-05 engine-speed SVG assets (dark theme, matches
gen_grid_chart.py). All values verbatim from committed records:

  speed-progression : grid-sweep-2026-11-05.json, execution-log-2026-11-05c.txt,
                      sweep-20261105T113659Z.json
  phase1-verdicts   : sweep-20261105T104604Z.json, T121018Z/T121117Z/T121215Z
  lab-calibration   : research/cuda-lab/reports/lab-20261105T113806Z.md
Metric classes (best-observed / median / sustained) are kept visually
distinct and never mixed.
"""

BG, PANEL, GRID = '#0b1017', '#101826', '#1d2a3a'
C_A, C_B = '#00d4ff', '#8a5cf6'          # 5090 cyan, violet
C_DIM = '#155a70'                        # dim cyan (older/historical bars)
C_GREEN = '#34d399'                      # sustained real-run / goal
TXT, SUB = '#e8f1f8', '#8fa3b8'
FONT = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
FONT2 = 'ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial'

parts = []

# ---------------------------------------------------------------------------
# Chart 1 — speed progression
# ---------------------------------------------------------------------------
W, H = 900, 470
ML, MR, MT, MB = 64, 16, 78, 70
PW, PH = W - ML - MR, H - MT - MB
YMAX = 10500.0

def y(v):
    return MT + PH - (v / YMAX) * PH

bars = [
    ("initial baseline\ndefault config", 7800.4, C_DIM,  "best observed"),
    ("grid config 512,64\n(geometry only)", 8133.0, C_DIM,  "best observed"),
    ("REAL SEARCH RUN\n512,64 · full 22 s", 8534.1, C_GREEN, "sustained"),
    ("median of 5 reps\n#39+#40 code", 9040.3, C_A, "median"),
    ("best observed\nsame build", 9681.5, C_A, "best observed"),
]

parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Engine speed progression on RTX 5090, 2026-11-05">')
parts.append(f'<rect width="{W}" height="{H}" rx="12" fill="{BG}"/>')
parts.append(f'<text x="{W/2}" y="28" text-anchor="middle" font-family="{FONT2}" font-size="16" font-weight="700" fill="{TXT}">Engine Speed Progression — RTX 5090 (Mkeys/s)</text>')
parts.append(f'<text x="{W/2}" y="48" text-anchor="middle" font-family="{FONT}" font-size="10" fill="{SUB}">2026-11-05 · CUDA 13.4 · every value traceable to a committed record</text>')

# legend (metric classes)
lx = ML
for label, col, outline in [("best observed", C_DIM, False), ("median of 5 reps", C_A, False), ("real-run sustained", C_GREEN, False)]:
    parts.append(f'<rect x="{lx}" y="58" width="10" height="10" rx="2" fill="{col}"/>')
    parts.append(f'<text x="{lx+16}" y="67" font-family="{FONT2}" font-size="11" fill="{SUB}">{label}</text>')
    lx += 26 + 14 + len(label) * 5.6

# gridlines
for gv in range(7000, 10501, 500):
    yy = y(gv)
    lw = '1.5' if gv == 10000 else '1'
    dash = ' stroke-dasharray="6,5"' if gv == 10000 else ''
    col = C_GREEN if gv == 10000 else GRID
    parts.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W-MR}" y2="{yy:.1f}" stroke="{col}" stroke-width="{lw}"{dash}/>')
    if gv != 10000:
        parts.append(f'<text x="{ML-8}" y="{yy+4:.1f}" text-anchor="end" font-family="{FONT}" font-size="10" fill="{SUB}">{gv}</text>')
parts.append(f'<text x="{ML+6}" y="{y(10000)-8:.1f}" font-family="{FONT2}" font-size="11" font-weight="700" fill="{C_GREEN}">goal: >10,000</text>')
parts.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="{GRID}" stroke-width="1"/>')

gw = PW / len(bars)
bw = 74
for i, (label, v, col, mclass) in enumerate(bars):
    cx = ML + gw * (i + 0.5)
    xa = cx - bw / 2
    hh = (MT + PH) - y(v)
    stroke = ' stroke="#ffffff" stroke-width="1.5"' if v == 9681.5 else ''
    parts.append(f'<rect x="{xa:.1f}" y="{y(v):.1f}" width="{bw}" height="{hh:.1f}" rx="3" fill="{col}"{stroke}></rect>')
    parts.append(f'<text x="{cx:.1f}" y="{y(v)-6:.1f}" text-anchor="middle" font-family="{FONT}" font-size="11" fill="{TXT}">{v:,.1f}</text>')
    l1, l2 = label.split("\n")
    parts.append(f'<text x="{cx:.1f}" y="{MT+PH+18:.1f}" text-anchor="middle" font-family="{FONT}" font-size="9" fill="{TXT}">{l1}</text>')
    parts.append(f'<text x="{cx:.1f}" y="{MT+PH+30:.1f}" text-anchor="middle" font-family="{FONT}" font-size="9" fill="{SUB}">{l2}</text>')

parts.append(f'<text x="{W/2}" y="{H-28}" text-anchor="middle" font-family="{FONT2}" font-size="10" fill="{SUB}">metric classes are reported separately and never mixed · white outline = current best</text>')
parts.append(f'<text x="{W/2}" y="{H-12}" text-anchor="middle" font-family="{FONT2}" font-size="10" fill="{SUB}">+19.8% real-run vs historical (7,128.4 same test vector) · +6.3% median code gain (noise floor 0.5%)</text>')
parts.append('</svg>')
open('docs/assets/speed-progression.svg', 'w').write('\n'.join(parts))
print("written: speed-progression.svg")
parts = []

# ---------------------------------------------------------------------------
# Chart 2 — Phase-1 verdicts (two panels)
# ---------------------------------------------------------------------------
W, H = 900, 440
BG_Y0 = 80

parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Phase-1 experiment verdicts: geometry plateau and null build levers">')
parts.append(f'<rect width="{W}" height="{H}" rx="12" fill="{BG}"/>')
parts.append(f'<text x="{W/2}" y="26" text-anchor="middle" font-family="{FONT2}" font-size="15" font-weight="700" fill="{TXT}">Phase-1 Zero-Risk Levers — All Measured, All Closed (RTX 5090)</text>')
parts.append(f'<text x="{W/2}" y="44" text-anchor="middle" font-family="{FONT}" font-size="9.5" fill="{SUB}">medians of 5 reps · same day back-to-back · 2026-11-05 · noise floor ±0.5%</text>')

# ---- left panel: E-S1 geometry (medians, commit 07a8509)
LX, LW = 24, 430
LT, LB = 66, 120          # top/bottom of plot area (y coords)
PH1 = LB - LT
gmin, gmax = 7600.0, 8700.0

def gy(v):
    return LB - (v - gmin) / (gmax - gmin) * PH1

es1 = [("128,16", 7848.8), ("256,32", 8289.9), ("512,16", 8544.9), ("512,32", 8551.2), ("512,64", 8546.0), ("1024,32", 8547.4)]
parts.append(f'<text x="{LX+LW/2}" y="{LT-12}" text-anchor="middle" font-family="{FONT2}" font-size="12" font-weight="700" fill="{TXT}">E-S1 · launch geometry (medians)</text>')
for gv in range(7600, 8701, 200):
    yy = gy(gv)
    parts.append(f'<line x1="{LX+34}" y1="{yy:.1f}" x2="{LX+LW}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
    parts.append(f'<text x="{LX+28}" y="{yy+3:.1f}" text-anchor="end" font-family="{FONT}" font-size="8.5" fill="{SUB}">{gv}</text>')
parts.append(f'<line x1="{LX+34}" y1="{LT}" x2="{LX+34}" y2="{LB}" stroke="{GRID}" stroke-width="1"/>')
# plateau band 8545-8551
parts.append(f'<rect x="{LX+34}" y="{gy(8551):.1f}" width="{LW-34}" height="{gy(8545)-gy(8551):.2f}" fill="{C_A}" fill-opacity="0.08" stroke="{C_A}" stroke-opacity="0.35" stroke-dasharray="3,3"/>')
parts.append(f'<text x="{LX+LW}" y="{gy(8551)-5:.1f}" text-anchor="end" font-family="{FONT2}" font-size="9.5" fill="{C_A}">batch ≥ 512: plateau 8,545–8,551 (±0.1%)</text>')
gw1 = (LW - 34) / len(es1)
bw1 = 34
for i, (g, v) in enumerate(es1):
    cx = LX + 34 + gw1 * (i + 0.5)
    best = (v == 8551.2)
    stroke = ' stroke="#ffffff" stroke-width="1.2"' if best else ''
    parts.append(f'<rect x="{cx-bw1/2:.1f}" y="{gy(v):.1f}" width="{bw1}" height="{LB-gy(v):.1f}" rx="2" fill="{C_A}"{stroke}></rect>')
    parts.append(f'<text x="{cx:.1f}" y="{gy(v)-4:.1f}" text-anchor="middle" font-family="{FONT}" font-size="8" fill="{TXT}">{v:.0f}</text>')
    parts.append(f'<text x="{cx:.1f}" y="{LB+14:.1f}" text-anchor="middle" font-family="{FONT}" font-size="8.5" fill="{SUB}">{g}</text>')
parts.append(f'<text x="{LX+LW/2}" y="{LB+32}" text-anchor="middle" font-family="{FONT2}" font-size="9.5" fill="{SUB}">verdict: geometry saturated — no path to >10 G from launch tuning</text>')

# ---- right panel: E-S2 / E-S4 build levers (medians, commit 5c78d872)
RX, RW = 470, 406
RT, RB = 66, 120
PH2 = RB - RT
vmin, vmax = 8950.0, 9100.0

def ry(v):
    return RB - (v - vmin) / (vmax - vmin) * PH2

es2 = [("default build", 9006.7, C_A), ("RDC=0", 9041.0, C_B), ("MAX_BATCH=512", 9046.0, C_B)]
parts.append(f'<text x="{RX+RW/2}" y="{RT-12}" text-anchor="middle" font-family="{FONT2}" font-size="12" font-weight="700" fill="{TXT}">E-S2 / E-S4 · build levers (medians, axis zoomed)</text>')
# noise band ±0.5% around default
band_top, band_bot = ry(9006.7 * 1.005), ry(9006.7 * 0.995)
parts.append(f'<rect x="{RX+8}" y="{band_top:.1f}" width="{RW-8}" height="{band_bot-band_top:.1f}" fill="{C_GREEN}" fill-opacity="0.10" stroke="{C_GREEN}" stroke-opacity="0.4" stroke-dasharray="4,4"/>')
parts.append(f'<text x="{RX+RW-4}" y="{band_top-5:.1f}" text-anchor="end" font-family="{FONT2}" font-size="9.5" fill="{C_GREEN}">±0.5% same-day noise band</text>')
for gv in range(8950, 9101, 50):
    yy = ry(gv)
    parts.append(f'<line x1="{RX+70}" y1="{yy:.1f}" x2="{RX+RW}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
    parts.append(f'<text x="{RX+64}" y="{yy+3:.1f}" text-anchor="end" font-family="{FONT}" font-size="8.5" fill="{SUB}">{gv}</text>')
parts.append(f'<line x1="{RX+70}" y1="{RT}" x2="{RX+70}" y2="{RB}" stroke="{GRID}" stroke-width="1"/>')
gw2 = (RW - 70) / len(es2)
bw2 = 56
for i, (g, v, col) in enumerate(es2):
    cx = RX + 70 + gw2 * (i + 0.5)
    parts.append(f'<rect x="{cx-bw2/2:.1f}" y="{ry(v):.1f}" width="{bw2}" height="{RB-ry(v):.1f}" rx="2" fill="{col}"></rect>')
    parts.append(f'<text x="{cx:.1f}" y="{ry(v)-4:.1f}" text-anchor="middle" font-family="{FONT}" font-size="8.5" fill="{TXT}">{v:,.1f}</text>')
    parts.append(f'<text x="{cx:.1f}" y="{RB+14:.1f}" text-anchor="middle" font-family="{FONT}" font-size="8.5" fill="{SUB}">{g}</text>')
parts.append(f'<text x="{RX+RW/2}" y="{RB+32}" text-anchor="middle" font-family="{FONT2}" font-size="9.5" fill="{SUB}">verdict: +0.38% / +0.44% — inside noise → NULL (recorded, not hidden)</text>')
parts.append(f'<text x="{W/2}" y="{H-64}" text-anchor="middle" font-family="{FONT2}" font-size="10" fill="{SUB}">Phase 1 closed: no zero-risk lever moves this kernel — the remaining +11% to 10 Gkeys/s is kernel work (K-2 hash → K-1 subp → K-3 point-add)</text>')
parts.append(f'<text x="{W/2}" y="{H-46}" text-anchor="middle" font-family="{FONT}" font-size="9.5" fill="{SUB}">data: sweep-20261105T104604Z.json · sweep-20261105T121018Z / T121117Z / T121215Z.json</text>')
parts.append('</svg>')
open('docs/assets/phase1-verdicts.svg', 'w').write('\n'.join(parts))
print("written: phase1-verdicts.svg")
parts = []

# ---------------------------------------------------------------------------
# Chart 3 — lab calibration (where the ceiling is)
# ---------------------------------------------------------------------------
W, H = 900, 330

parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Lab calibration: modmul ceiling versus engine usage, RTX 5090">')
parts.append(f'<rect width="{W}" height="{H}" rx="12" fill="{BG}"/>')
parts.append(f'<text x="{W/2}" y="26" text-anchor="middle" font-family="{FONT2}" font-size="15" font-weight="700" fill="{TXT}">Where the Ceiling Is — Lab Calibration (RTX 5090)</text>')
parts.append(f'<text x="{W/2}" y="44" text-anchor="middle" font-family="{FONT}" font-size="9.5" fill="{SUB}">secp256k1 256-bit modular multiplications per second, same unit on both bars</text>')

BX, BW = 30, 520          # bar area
BY1, BY2, BH = 86, 142, 26
CEIL = 95.42

def bx(v):
    return BX + (v / CEIL) * BW

# ceiling bar
parts.append(f'<rect x="{BX}" y="{BY1}" width="{BW}" height="{BH}" rx="3" fill="{C_A}" fill-opacity="0.85"></rect>')
parts.append(f'<text x="{BX+10}" y="{BY1+17}" font-family="{FONT}" font-size="12" fill="#04222b" font-weight="700">95.42 Gops/s</text>')
parts.append(f'<text x="{W-16}" y="{BY1+17}" text-anchor="end" font-family="{FONT2}" font-size="11" fill="{TXT}">modmul256 ceiling — lab bench (median of 5)</text>')
# engine bar
parts.append(f'<rect x="{BX}" y="{BY2}" width="{bx(31.6):.1f}" height="{BH}" rx="3" fill="{C_B}"></rect>')
parts.append(f'<text x="{BX+10}" y="{BY2+17}" font-family="{FONT}" font-size="12" fill="#120726" font-weight="700">31.6</text>')
parts.append(f'<text x="{W-16}" y="{BY2+17}" text-anchor="end" font-family="{FONT2}" font-size="11" fill="{TXT}">engine at 9,040 Mkeys/s — derived (3.5 modmul/key)</text>')
# 33% annotation
parts.append(f'<line x1="{bx(31.6):.1f}" y1="{BY1}" x2="{bx(31.6):.1f}" y2="{BY2+BH}" stroke="{C_B}" stroke-width="1" stroke-dasharray="4,3"/>')
parts.append(f'<text x="{bx(31.6)+8:.1f}" y="{(BY1+BY2+BH)/2:.1f}" font-family="{FONT2}" font-size="11" fill="{C_B}">33% of ceiling</text>')

# stat callouts
stats = [
    ("EC arithmetic", "3.0x ALU headroom — NOT the bottleneck"),
    ("instruction issue", "~4,000 instr/key vs ~44 T/s peak  →  wall ≈ 11 Gkeys/s (est.)"),
    ("power", "64–66 W of 575 W — not power/thermal bound"),
    ("verdict", "path to >10 Gkeys/s = fewer instructions per key (hash160 first)"),
]
sy = 196
for k, v in stats:
    parts.append(f'<text x="{BX}" y="{sy}" font-family="{FONT}" font-size="9.5" fill="{C_A}">{k}</text>')
    parts.append(f'<text x="{BX+130}" y="{sy}" font-family="{FONT2}" font-size="10.5" fill="{SUB}">{v}</text>')
    sy += 24

parts.append(f'<text x="{W/2}" y="{H-12}" text-anchor="middle" font-family="{FONT2}" font-size="9.5" fill="{SUB}">data: research/cuda-lab/reports/lab-20261105T113806Z.md — synthetic Gops never compared with engine Mkeys/s; only same-unit ratios are used</text>')
parts.append('</svg>')
open('docs/assets/lab-calibration.svg', 'w').write('\n'.join(parts))
print("written: lab-calibration.svg")
