"""Generate docs/assets/grid-comparison.svg — grouped bar chart of the
2026-11-05 grid sweep (RTX 5090 vs RTX PRO 6000), values verbatim from
docs/benchmarks/data/grid-sweep-2026-11-05.json. Missing data stays
missing (N/A marker), never zero."""

import json

DATA = json.load(open('docs/benchmarks/data/grid-sweep-2026-11-05.json'))
grids = list(DATA['grid_sweep_mkeys_per_s'].keys())
a = [DATA['grid_sweep_mkeys_per_s'][g]['rtx_5090'] for g in grids]
b = [DATA['grid_sweep_mkeys_per_s'][g]['rtx_pro_6000'] for g in grids]

W, H = 900, 420
ML, MR, MT, MB = 64, 16, 74, 66   # margins
PW, PH = W - ML - MR, H - MT - MB
YMAX = 8500.0

def y(v):
    return MT + PH - (v / YMAX) * PH

BG, PANEL, GRID = '#0b1017', '#101826', '#1d2a3a'
C_A, C_B = '#00d4ff', '#8a5cf6'          # 5090 cyan, PRO 6000 violet
TXT, SUB = '#e8f1f8', '#8fa3b8'
FONT = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
FONT2 = 'ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial'

parts = []
parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Grid sweep comparison: RTX 5090 vs RTX PRO 6000">')
parts.append(f'<rect width="{W}" height="{H}" rx="12" fill="{BG}"/>')
parts.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="{FONT2}" font-size="16" font-weight="700" fill="{TXT}">Grid Sweep — Best Observed Throughput (Mkeys/s)</text>')
parts.append(f'<text x="{W/2}" y="50" text-anchor="middle" font-family="{FONT}" font-size="10" fill="{SUB}">early-termination known-answer runs · commit b3d13be · CUDA 13.4 · 2026-11-05</text>')

# legend
parts.append(f'<rect x="{ML}" y="58" width="10" height="10" fill="{C_A}"/>')
parts.append(f'<text x="{ML+16}" y="67" font-family="{FONT2}" font-size="11" fill="{SUB}">GeForce RTX 5090 (170 SM)</text>')
parts.append(f'<rect x="{ML+210}" y="58" width="10" height="10" fill="{C_B}"/>')
parts.append(f'<text x="{ML+226}" y="67" font-family="{FONT2}" font-size="11" fill="{SUB}">RTX PRO 6000 Blackwell (188 SM)</text>')

# y gridlines + labels
for gv in range(0, 8501, 1000):
    yy = y(gv)
    parts.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{W-MR}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
    parts.append(f'<text x="{ML-8}" y="{yy+4:.1f}" text-anchor="end" font-family="{FONT}" font-size="10" fill="{SUB}">{gv}</text>')
parts.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="{GRID}" stroke-width="1"/>')

gw = PW / len(grids)
bw = 22
best_a, best_b = 8133.0, 7452.4

for i, g in enumerate(grids):
    cx = ML + gw * (i + 0.5)
    xa = cx - bw - 2
    xb = cx + 2
    va, vb = a[i], b[i]
    is_best_a = (va == best_a)
    is_best_b = (vb == best_b)
    # bars
    if va is None:
        parts.append(f'<text x="{cx:.1f}" y="{MT+PH-14:.1f}" text-anchor="middle" font-family="{FONT}" font-size="9" fill="{SUB}">N/A</text>')
    else:
        hh = (MT + PH) - y(va)
        parts.append(f'<rect x="{xa:.1f}" y="{y(va):.1f}" width="{bw}" height="{hh:.1f}" rx="2" fill="{C_A}"{"" if not is_best_a else ' stroke="#ffffff" stroke-width="1.5"'}>')
        parts.append('</rect>')
        parts.append(f'<text x="{xa+bw/2:.1f}" y="{y(va)-4:.1f}" text-anchor="middle" font-family="{FONT}" font-size="7.5" fill="{TXT}">{va:.0f}</text>')
    if vb is None:
        pass  # single N/A marker covers both
    else:
        hh = (MT + PH) - y(vb)
        parts.append(f'<rect x="{xb:.1f}" y="{y(vb):.1f}" width="{bw}" height="{hh:.1f}" rx="2" fill="{C_B}"{"" if not is_best_b else ' stroke="#ffffff" stroke-width="1.5"'}>')
        parts.append('</rect>')
        parts.append(f'<text x="{xb+bw/2:.1f}" y="{y(vb)-4:.1f}" text-anchor="middle" font-family="{FONT}" font-size="7.5" fill="{TXT}">{vb:.0f}</text>')
    # x label
    parts.append(f'<text x="{cx:.1f}" y="{MT+PH+16:.1f}" text-anchor="middle" font-family="{FONT}" font-size="9.5" fill="{SUB}">{g}</text>')

parts.append(f'<text x="{W/2}" y="{H-26}" text-anchor="middle" font-family="{FONT2}" font-size="10" fill="{SUB}">--grid: points-per-batch, batches-per-SM · white outline = per-GPU best</text>')
parts.append(f'<text x="{W/2}" y="{H-10}" text-anchor="middle" font-family="{FONT2}" font-size="10" fill="{SUB}">N/A = reference key found, no speed line produced (not zero)</text>')
parts.append('</svg>')

open('docs/assets/grid-comparison.svg', 'w').write('\n'.join(parts))
print("written: docs/assets/grid-comparison.svg,", len(grids), "grid groups")
