#!/usr/bin/env python3
"""Render campaign_status.sh JSON into a standalone HTML dashboard.

    bash scripts/transformer_bench/campaign_status.sh HOST:PORT > status.json
    python3 scripts/transformer_bench/render_dashboard.py status.json > campaign.html

A SNAPSHOT, NOT A LIVE VIEW. The page has no network access once published (the artifact CSP blocks
every external request), so the numbers are frozen at render time and the header says so. For a
genuinely live view use monitor_campaign.sh in a terminal; this exists to be readable and shareable.

EXAMM COMES FROM examm_cells.json, NOT FROM THE POD. There are no EXAMM runs in this campaign -- it
is the incumbent, already measured at 10 seeds per cell, and its numbers are lifted from the paper
table with provenance recorded in that file. The two sources are comparable because both score the
same windows with the same Algorithm 2 at L/S 10, and because EXAMM's GROSS column is used:
score_cells.py applies no transaction cost, so pairing it against a net figure would flatter the
transformers by roughly the spread between the two.

CROSSFORMER IS EXCLUDED. At the authors' Traffic configuration it collapses to a near-constant
prediction on this universe and opens Algorithm 2's gate on 0 of 249 days in every cell, so its
return is undefined rather than zero. Dropping it from the comparison is what the paper does; the
page still says so rather than quietly showing one model fewer.

The scatter is the point of the page. IC and trading return dissociate on this study -- Algorithm 2
only opens a position when all top-L predictions are >0 and all bottom-S are <0, so ranking skill
and realised return come apart, and a table of two number columns hides that while a scatter makes
it immediate.
"""
from __future__ import annotations

import json
import os
import sys
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = {"CrossformerOfficial"}
# The 12-cell zero-shot grid is scored LOCALLY (results/zeroshot), not on the pod, so it is merged
# in separately. Only moirai20Rsmall is shown: it is the complete, deterministic 12-cell result.
# The moirai11Rbase/_med/S500 entries are the single-cell probe on set1/2022 -- the one cell of
# twelve that favours momentum -- and putting a cherry-picked cell beside 12-cell rows would
# misrepresent it.
ZEROSHOT = {"moirai20Rsmall"}
SHORT = {"PatchTSTOfficial": "PatchTST", "ITransformerOfficial": "iTransformer",
         "CrossformerOfficial": "Crossformer", "DeformTime": "DeformTime",
         "DLinearOfficial": "DLinear", "EXAMM": "EXAMM", "moirai20Rsmall": "MOIRAI 2.0",
         "TimeMixer": "TimeMixer"}
PAL = {"EXAMM": "var(--accent)", "PatchTSTOfficial": "#7C8CF0",
       "ITransformerOfficial": "var(--pos)", "DeformTime": "#C77DBB",
       "DLinearOfficial": "var(--flat)", "moirai20Rsmall": "#3FA8A0",
       "TimeMixer": "#E0A458"}
ORDER = ["EXAMM", "PatchTSTOfficial", "ITransformerOfficial", "DeformTime", "DLinearOfficial",
         "TimeMixer", "moirai20Rsmall"]

CSS = """
:root{
  --bg:#F4F5F7; --panel:#FFFFFF; --ink:#171A21; --dim:#6B7280; --line:#E2E5EA;
  --accent:#B4762A; --pos:#2A7D5F; --neg:#B23F38; --flat:#8A8FA0;
  --bar:#CBD1DA; --barfill:#171A21; --bestbg:rgba(180,118,42,.13);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0E1014; --panel:#161A21; --ink:#E8EAEE; --dim:#8A92A3; --line:#252A33;
    --accent:#D9A04A; --pos:#4FB58B; --neg:#E0716A; --flat:#6E7688;
    --bar:#2A303A; --barfill:#D9A04A; --bestbg:rgba(217,160,74,.16);
  }
}
:root[data-theme="dark"]{
  --bg:#0E1014; --panel:#161A21; --ink:#E8EAEE; --dim:#8A92A3; --line:#252A33;
  --accent:#D9A04A; --pos:#4FB58B; --neg:#E0716A; --flat:#6E7688;
  --bar:#2A303A; --barfill:#D9A04A; --bestbg:rgba(217,160,74,.16);
}
:root[data-theme="light"]{
  --bg:#F4F5F7; --panel:#FFFFFF; --ink:#171A21; --dim:#6B7280; --line:#E2E5EA;
  --accent:#B4762A; --pos:#2A7D5F; --neg:#B23F38; --flat:#8A8FA0;
  --bar:#CBD1DA; --barfill:#171A21; --bestbg:rgba(180,118,42,.13);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-variant-numeric:tabular-nums; line-height:1.5;
}
.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
.wrap{max-width:1120px; margin:0 auto; padding:32px 20px 64px; display:flex; flex-direction:column; gap:24px}
header{display:flex; flex-direction:column; gap:6px; border-bottom:1px solid var(--line); padding-bottom:18px}
h1{margin:0; font-size:1.5rem; font-weight:650; letter-spacing:-0.02em}
.eyebrow{font-size:.68rem; text-transform:uppercase; letter-spacing:.14em; color:var(--accent); font-weight:600}
.sub{color:var(--dim); font-size:.86rem}
.strip{display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:1px; background:var(--line);
  border:1px solid var(--line); border-radius:8px; overflow:hidden}
.stat{background:var(--panel); padding:12px 14px; display:flex; flex-direction:column; gap:2px}
.stat .k{font-size:.66rem; text-transform:uppercase; letter-spacing:.1em; color:var(--dim)}
.stat .v{font-size:1.18rem; font-weight:600}
.panel{background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px 18px}
h2{margin:0 0 4px; font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color:var(--dim); font-weight:600}
.h2sub{font-size:.78rem; color:var(--dim); margin-bottom:12px}
.arm{display:grid; grid-template-columns:150px 1fr 74px; align-items:center; gap:12px; padding:5px 0}
.track{height:7px; background:var(--bar); border-radius:4px; overflow:hidden}
.fill{height:100%; background:var(--barfill); border-radius:4px}
.fill.done{background:var(--pos)}
.count{font-size:.8rem; color:var(--dim); text-align:right}
.cols{display:grid; grid-template-columns:1fr 1fr; gap:16px}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
.row{display:flex; justify-content:space-between; gap:10px; padding:3px 0; font-size:.8rem}
.row .lab{white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.row .val{color:var(--dim); white-space:nowrap}
.epbar{display:inline-block; height:6px; background:var(--accent); border-radius:3px; vertical-align:middle; margin-right:6px}
.scroll{overflow-x:auto}
table{border-collapse:collapse; width:100%; font-size:.82rem}
th{text-align:right; font-size:.64rem; text-transform:uppercase; letter-spacing:.08em; color:var(--dim);
   font-weight:600; padding:0 7px 7px; border-bottom:1px solid var(--line); white-space:nowrap}
th.grp{text-align:center; padding-bottom:3px; color:var(--ink); border-bottom:0}
th:first-child,td:first-child{text-align:left}
td{padding:5px 7px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap}
tr.mh td{padding-top:14px; font-weight:600; font-size:.76rem; text-transform:uppercase;
  letter-spacing:.07em; color:var(--accent)}
tr.grpend td{border-bottom:2px solid var(--line)}
.sep{border-left:1px solid var(--line)}
.pos{color:var(--pos)} .neg{color:var(--neg)} .flat{color:var(--flat)}
.best{background:var(--bestbg); font-weight:700; border-radius:3px}
sup.n{font-size:.6rem; color:var(--dim); font-weight:400; margin-left:1px}
.chip{display:inline-block; padding:1px 6px; border-radius:9px; font-size:.62rem; font-weight:600;
  letter-spacing:.05em; text-transform:uppercase; background:var(--neg); color:#fff}
.gate{display:inline-block; width:42px; height:5px; background:var(--bar); border-radius:3px; vertical-align:middle}
.gate i{display:block; height:100%; background:var(--flat); border-radius:3px}
.note{font-size:.78rem; color:var(--dim); border-left:2px solid var(--accent); padding:2px 0 2px 12px}
.legend{display:flex; gap:14px; flex-wrap:wrap; font-size:.72rem; color:var(--dim); margin-top:8px}
.dot{display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; vertical-align:middle}
svg{display:block; width:100%; height:auto}
"""


def cls(v):
    return "pos" if v > 0 else ("neg" if v < 0 else "flat")


def scatter(pts, colors):
    """Ensemble rank IC (x) against Algorithm 2 gross return (y), one dot per cell."""
    if not pts:
        return ""
    W, H, P = 640, 320, 46
    xs = [c["ic"] for c in pts]
    ys = [c["ret"] for c in pts]
    x0, x1 = min(xs + [0]), max(xs + [0])
    y0, y1 = min(ys + [0]), max(ys + [0])
    px = (x1 - x0) * 0.12 or 0.01
    py = (y1 - y0) * 0.12 or 1
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    sx = lambda v: P + (v - x0) / (x1 - x0) * (W - P - 12)
    sy = lambda v: H - P - (v - y0) / (y1 - y0) * (H - P - 14)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Rank IC against trading return by cell">']
    # zero axes: the quadrants are the message -- upper-left and lower-right hold cells where
    # ranking skill and realised return disagree in sign.
    o.append(f'<line x1="{sx(0):.1f}" y1="14" x2="{sx(0):.1f}" y2="{H-P}" stroke="var(--line)"/>')
    o.append(f'<line x1="{P}" y1="{sy(0):.1f}" x2="{W-12}" y2="{sy(0):.1f}" stroke="var(--line)"/>')
    for v in (x0, 0, x1):
        o.append(f'<text x="{sx(v):.1f}" y="{H-P+16}" fill="var(--dim)" font-size="9" '
                 f'text-anchor="middle" font-family="ui-monospace,monospace">{v:+.3f}</text>')
    for v in (y0, 0, y1):
        o.append(f'<text x="{P-6}" y="{sy(v)+3:.1f}" fill="var(--dim)" font-size="9" '
                 f'text-anchor="end" font-family="ui-monospace,monospace">{v:+.0f}%</text>')
    o.append(f'<text x="{(W+P)/2:.0f}" y="{H-8}" fill="var(--dim)" font-size="10" '
             f'text-anchor="middle">ensemble rank IC</text>')
    o.append(f'<text x="13" y="{H/2:.0f}" fill="var(--dim)" font-size="10" text-anchor="middle" '
             f'transform="rotate(-90 13 {H/2:.0f})">Algorithm 2 gross return</text>')
    for c in pts:
        col = colors.get(c["model"], "var(--flat)")
        r = 3.2 + min(c.get("n", 10), 10) * 0.28
        o.append(f'<circle cx="{sx(c["ic"]):.1f}" cy="{sy(c["ret"]):.1f}" r="{r:.1f}" fill="{col}" '
                 f'fill-opacity="0.7" stroke="{col}"><title>'
                 f'{escape(SHORT.get(c["model"], c["model"]))} {c["set"]}/{c["yr"]} '
                 f'n={c.get("n",10)} IC {c["ic"]:+.4f} ret {c["ret"]:+.2f}%</title></circle>')
    o.append("</svg>")
    return "".join(o)


def matrix(by_model, examm, models):
    """Paper-style 4-draw x 3-year matrix: IC and gross return per model, best in each row tinted.

    Best is computed per row and per metric across the models present. A cell whose ensemble is
    still short of 10 seeds carries its seed count as a superscript, because an ensemble of 2 is
    not a like-for-like opponent to one of 10 and the highlight alone would not say so.
    """
    bench = {(c["set"], c["yr"]): c for c in examm}
    sets = sorted({c["set"] for c in examm})
    years = sorted({c["yr"] for c in examm})
    o = ['<div class="scroll"><table>']
    o.append('<thead><tr><th colspan="2"></th>'
             f'<th class="grp sep" colspan="{len(models)}">Rank IC</th>'
             f'<th class="grp sep" colspan="{len(models)}">Gross return (%)</th>'
             '<th class="grp sep" colspan="2">Benchmark (%)</th></tr><tr>'
             '<th>Draw</th><th>Trade yr</th>')
    for grp in range(2):
        for i, m in enumerate(models):
            o.append(f'<th class="{"sep" if i==0 else ""}">{escape(SHORT.get(m,m))}</th>')
        _ = grp
    o.append('<th class="sep">EW B&amp;H</th><th>S&amp;P 500</th></tr></thead><tbody>')

    for si, st in enumerate(sets):
        for yi, yr in enumerate(years):
            vals = {m: by_model.get(m, {}).get((st, yr)) for m in models}
            b = bench.get((st, yr), {})
            end = " grpend" if yi == len(years) - 1 and si < len(sets) - 1 else ""
            o.append(f'<tr class="{end.strip()}"><td class="mono">{st}</td><td class="mono">{yr}</td>')
            for key, fmt in (("ic", "{:+.4f}"), ("ret", "{:+.2f}")):
                present = {m: v[key] for m, v in vals.items() if v and v.get(key) is not None}
                top = max(present.values()) if present else None
                for i, m in enumerate(models):
                    v = vals.get(m)
                    sep = " sep" if i == 0 else ""
                    if not v or v.get(key) is None:
                        o.append(f'<td class="flat{sep}">—</td>')
                        continue
                    x = v[key]
                    best = " best" if top is not None and x == top else ""
                    n = v.get("n", 10)
                    sup = f'<sup class="n">{n}</sup>' if n < 10 else ""
                    o.append(f'<td class="mono {cls(x)}{best}{sep}">{fmt.format(x)}{sup}</td>')
            o.append(f'<td class="mono flat sep">{b.get("ew",float("nan")):+.2f}</td>'
                     f'<td class="mono flat">{b.get("sp500",float("nan")):+.2f}</td></tr>')

    # column means, over the cells each model actually has -- stated on the row, since a model
    # part-way through the campaign is not averaging the same 12 cells as a finished one.
    o.append('<tr class="mh"><td colspan="2">Mean</td>')
    for key, fmt in (("ic", "{:+.4f}"), ("ret", "{:+.2f}")):
        means = {}
        for m in models:
            vs = [v[key] for v in by_model.get(m, {}).values() if v.get(key) is not None]
            if vs:
                means[m] = sum(vs) / len(vs)
        top = max(means.values()) if means else None
        for i, m in enumerate(models):
            sep = " sep" if i == 0 else ""
            if m not in means:
                o.append(f'<td class="flat{sep}">—</td>')
                continue
            cnt = len([v for v in by_model.get(m, {}).values() if v.get(key) is not None])
            best = " best" if means[m] == top else ""
            sup = f'<sup class="n">{cnt}c</sup>' if cnt < len(examm) else ""
            o.append(f'<td class="mono {cls(means[m])}{best}{sep}">{fmt.format(means[m])}{sup}</td>')
    ew = sum(c["ew"] for c in examm) / len(examm)
    sp = sum(c["sp500"] for c in examm) / len(examm)
    o.append(f'<td class="mono flat sep">{ew:+.2f}</td><td class="mono flat">{sp:+.2f}</td></tr>')
    o.append("</tbody></table></div>")
    return "".join(o)


def main():
    d = json.load(open(sys.argv[1]))
    ex = json.load(open(os.path.join(HERE, "examm_cells.json")))["cells"]
    # MERGE THE LOCAL SCORE CACHE OVER THE POD'S. The status JSON carries only the pod it was
    # polled from, and the campaign now spans two: DeformTime on one, DLinear on the other. The
    # local cache is scored from results synced off BOTH, so it is the only complete view. Pod
    # status is still used for live progress (arms, in-flight, GPU) -- that genuinely is per-pod.
    local_cells = {}
    try:
        for r in json.load(open(os.path.join(
                os.path.dirname(HERE), "..",
                "results/transformer_bench/.cell_scores.json"))).values():
            if r.get("ic") is None:
                continue
            k = (r["model"], r["set"], r["yr"])
            if r["n"] >= local_cells.get(k, {}).get("n", -1):
                local_cells[k] = r
    except FileNotFoundError:
        pass
    if local_cells:
        keep = [c for c in d.get("cells", [])
                if (c["model"], c["set"], c["yr"]) not in local_cells]
        d["cells"] = keep + list(local_cells.values())

    # merge the locally-scored zero-shot grid; same score_cells.py path, same trader
    zs = []
    try:
        for r in json.load(open(os.path.join(
                os.path.dirname(HERE), "..", "results/zeroshot/.cell_scores.json"))).values():
            if r.get("model") in ZEROSHOT and r.get("ic") is not None:
                zs.append(r)
    except Exception:
        pass
    d.setdefault("cells", []).extend(zs)
    cells = [c for c in d.get("cells", []) if c["model"] not in EXCLUDE and c.get("ic") is not None]
    arms = [a for a in d.get("arms", []) if a["model"] not in ("Crossformer",)]
    running = [r for r in d.get("running", []) if r["model"] not in EXCLUDE]
    tot = sum(a["done"] for a in arms)
    cap = sum(a["total"] for a in arms)

    by_model = {"EXAMM": {(c["set"], c["yr"]): {"ic": c["ic"], "ret": c["ret"], "n": 10} for c in ex}}
    for c in cells:
        by_model.setdefault(c["model"], {})[(c["set"], c["yr"])] = {
            "ic": c["ic"], "ret": c.get("ret"), "n": c["n"]}
    models = [m for m in ORDER if m in by_model]

    H = []
    H.append("<title>EXAMM vs transformers — portfolio campaign</title>")
    H.append(f"<style>{CSS}</style>")
    H.append('<div class="wrap">')
    H.append('<header><div class="eyebrow">mid_highmid · 4 independent 50-stock draws × 3 cohorts</div>'
             '<h1>EXAMM vs transformers</h1>'
             f'<div class="sub">Snapshot of a running campaign — {tot} of {cap} transformer runs '
             'complete. EXAMM is the finished 10-seed incumbent; the transformer arms are still '
             'filling in. Numbers freeze at render time; the live view is '
             '<span class="mono">monitor_campaign.sh</span>.</div></header>')

    gpus = " · ".join(f"{g['util']}%" for g in d.get("gpu", [])) or "—"
    H.append('<div class="strip">')
    for k, v in [("runs done", f"{tot}/{cap}"), ("in flight", str(len(running))),
                 ("shards", str(d.get("shards", 0))), ("scored cells", str(len(cells))),
                 ("gpu util", gpus)]:
        H.append(f'<div class="stat"><span class="k">{k}</span><span class="v">{v}</span></div>')
    H.append("</div>")

    # ---- headline: the paper-shaped matrix
    H.append('<div class="panel"><h2>Per-draw results</h2>'
             '<div class="h2sub">Rank IC and Algorithm 2 gross return (L/S 10, annual %) for every '
             'draw × trade-year cell. The best model in each row is tinted, per metric. '
             'A superscript marks an ensemble still short of 10 seeds.</div>')
    H.append(matrix(by_model, ex, models))
    H.append('<div class="legend">')
    for m in models:
        H.append(f'<span><i class="dot" style="background:{PAL.get(m,"var(--flat)")}"></i>'
                 f'{escape(SHORT.get(m,m))}</span>')
    H.append('</div>')
    H.append('<div class="note" style="margin-top:12px">EXAMM is complete at 10 seeds in all 12 '
             'cells; the transformer arms are mid-campaign, so a tinted transformer cell backed by '
             'one or two seeds is a provisional lead, not a result. Returns are gross throughout — '
             'the transformer runs carry no transaction cost, so EXAMM\'s gross column is the '
             'like-for-like one. Crossformer is excluded: at the authors\' Traffic configuration it '
             'collapses to a near-constant prediction here and opens the gate on 0 of 249 days, '
             'making its return undefined rather than zero.</div></div>')

    # ---- dissociation
    pts = [{"model": "EXAMM", "set": c["set"], "yr": c["yr"], "ic": c["ic"], "ret": c["ret"], "n": 10}
           for c in ex]
    pts += [{"model": c["model"], "set": c["set"], "yr": c["yr"], "ic": c["ic"],
             "ret": c["ret"], "n": c["n"]} for c in cells if c.get("ret") is not None]
    H.append('<div class="panel"><h2>Ranking skill vs realised return</h2>'
             '<div class="h2sub">One dot per cell; dot size is seeds in the ensemble.</div>')
    H.append(scatter(pts, PAL))
    H.append('<div class="legend">')
    for m in sorted({p["model"] for p in pts}, key=lambda x: ORDER.index(x) if x in ORDER else 9):
        H.append(f'<span><i class="dot" style="background:{PAL.get(m,"var(--flat)")}"></i>'
                 f'{escape(SHORT.get(m,m))}</span>')
    H.append('</div>')
    H.append('<div class="note" style="margin-top:12px">Points off the diagonal are why both '
             'columns are reported. Algorithm 2 takes a position only on days when every top-10 '
             'prediction is positive and every bottom-10 is negative, so a cell can rank stocks '
             'well and still lose money — set4/2022 is EXAMM\'s best IC of the twelve '
             '(+0.0277) and returns −5.30%.</div></div>')

    # ---- live campaign detail
    H.append('<div class="panel"><h2>Transformer arms</h2>')
    for a in arms:
        f = a["done"] / a["total"]
        done = " done" if a["done"] == a["total"] else ""
        H.append(f'<div class="arm"><span>{escape(a["model"])}</span>'
                 f'<span class="track"><span class="fill{done}" style="width:{f*100:.1f}%"></span></span>'
                 f'<span class="count mono">{a["done"]}/{a["total"]}</span></div>')
    H.append("</div>")

    H.append('<div class="cols">')
    H.append('<div class="panel"><h2>In flight</h2>')
    for r in sorted(running, key=lambda x: -x["epochs"]):
        H.append(f'<div class="row"><span class="lab mono">{escape(SHORT.get(r["model"],r["model"]))} '
                 f'{r["set"]}/{r["yr"]} s{r["seed"]}</span>'
                 f'<span class="val"><i class="epbar" style="width:{min(r["epochs"],50)*1.6:.0f}px">'
                 f'</i>{r["epochs"]} ep</span></div>')
    H.append("</div>")
    H.append('<div class="panel"><h2>Recently completed</h2>')
    for r in [x for x in d.get("recent", []) if x["model"] not in EXCLUDE][:9]:
        H.append(f'<div class="row"><span class="lab mono">{escape(SHORT.get(r["model"],r["model"]))} '
                 f'{r["set"]}/{r["yr"]} s{r["seed"]}</span>'
                 f'<span class="val mono">{r["epochs"]} ep · {r["min"]:.1f} min</span></div>')
    H.append("</div></div>")

    if cells:
        H.append('<div class="panel"><h2>Scored transformer cells</h2>'
                 '<div class="h2sub">Seed-mean ensembles as they land. Dispersion and gate days sit '
                 'beside IC because a collapsed model can hold a respectable IC while the gate '
                 'rarely opens.</div><div class="scroll"><table>')
        H.append('<thead><tr><th>cell</th><th>seeds</th><th>rank IC</th><th>t</th>'
                 '<th>gross %</th><th>dispersion</th><th>gate days</th></tr></thead><tbody>')
        for m in [x for x in ORDER if x in {c["model"] for c in cells}]:
            rows = [c for c in cells if c["model"] == m]
            ics = [c["ic"] for c in rows]
            rets = [c["ret"] for c in rows if c.get("ret") is not None]
            summ = f"{len(rows)} cells · mean IC {sum(ics)/len(ics):+.4f}"
            if rets:
                summ += (f" · mean return {sum(rets)/len(rets):+.2f}% · "
                         f"{sum(1 for r in rets if r>0)}/{len(rets)} up")
            H.append(f'<tr class="mh"><td colspan="7">{escape(SHORT.get(m,m))} '
                     f'<span style="color:var(--dim);text-transform:none;letter-spacing:0;'
                     f'font-weight:400">— {summ}</span></td></tr>')
            for c in sorted(rows, key=lambda x: (x["set"] or "", x["yr"])):
                ret = c.get("ret")
                rs = (f'<span class="{cls(ret)}">{ret:+.2f}</span>' if ret is not None
                      else '<span class="flat">—</span>')
                collapsed = (c.get("sd") or 1) < 1e-4
                g = c["gate"] / c["days"] * 100 if c["days"] else 0
                H.append(f'<tr><td class="mono">{c["set"]}/{c["yr"]}</td><td>{c["n"]}</td>'
                         f'<td class="{cls(c["ic"])} mono">{c["ic"]:+.4f}</td>'
                         f'<td class="flat mono">{c["t"]:+.2f}</td><td class="mono">{rs}</td>'
                         f'<td class="mono flat">{c["sd"]:.6f}'
                         f'{" <span class=chip>collapsed</span>" if collapsed else ""}</td>'
                         f'<td class="mono flat"><span class="gate"><i style="width:{g:.0f}%">'
                         f'</i></span> {c["gate"]}/{c["days"]}</td></tr>')
        H.append("</tbody></table></div></div>")

    H.append('<div class="note">Single-seed trading return on this study has spanned −2.03% to '
             '+14.62% across two seeds of the same cell, so cell-level transformer numbers move '
             'until their ensembles fill. EXAMM rows are final.</div>')
    H.append("</div>")
    print("\n".join(H))


if __name__ == "__main__":
    main()
