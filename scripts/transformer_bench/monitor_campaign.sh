#!/bin/bash
# Live campaign dashboard. Redraws in place every REFRESH seconds until Ctrl-C.
#
#   bash scripts/transformer_bench/monitor_campaign.sh root@1.2.3.4:42486 [REMOTE_REPO]
#   REFRESH=15 bash scripts/transformer_bench/monitor_campaign.sh root@1.2.3.4:42486
#
# Reads campaign_status.sh, so the numbers here and in the status artifact cannot drift apart.
# A failed poll prints a warning and keeps the previous frame rather than clearing the screen --
# this link drops often enough that a monitor which blanks on every hiccup is unreadable.
set -uo pipefail
HOST=${1:?usage: monitor_campaign.sh USER@HOST[:PORT] [REMOTE_REPO]}
REMOTE=${2:-/workspace/EXAMM-Extended}
REFRESH=${REFRESH:-20}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
started=$(date +%s)

while true; do
  json=$(bash "$HERE/campaign_status.sh" "$HOST" "$REMOTE" 2>/dev/null || true)
  if [ -n "$json" ]; then
    frame=$(REFRESH="$REFRESH" ELAPSED=$(( $(date +%s) - started )) python3 - "$json" <<'PY'
import json,sys,os
d=json.loads(sys.argv[1]); el=int(os.environ["ELAPSED"])
B="\033[1m"; D="\033[2m"; G="\033[32m"; Y="\033[33m"; C="\033[36m"; RD="\033[31m"; R="\033[0m"
rtc=lambda v: G if v>5 else (Y if v>0 else RD)
L=[]
L.append(f"{B}  EXAMM-Extended  ·  portfolio campaign{R}   {D}elapsed {el//3600}h{el%3600//60:02d}m · refresh {os.environ['REFRESH']}s · Ctrl-C to exit{R}")
L.append("")
tot=sum(a['done'] for a in d['arms']); cap=sum(a['total'] for a in d['arms'])
for a in d['arms']:
    f=a['done']/a['total']; w=38; n=int(f*w)
    col = G if a['done']==a['total'] else (C if a['done'] else D)
    bar = col+"█"*n+R+D+"·"*(w-n)+R
    L.append(f"   {a['model']:<20s} {bar} {a['done']:>3d}/{a['total']}")
L.append("")
L.append(f"   {D}total {tot}/{cap} runs · {len(d.get('running',[]))} in flight · {d.get('shards',0)} shards{R}")
if d.get("gpu"):
    g="  ".join(f"gpu{x['i']} {Y if x['util']>50 else D}{x['util']:>3d}%{R}{D}/{x['mem']//1024}G{R}" for x in d["gpu"])
    L.append(f"   {g}")
L.append("")
if d.get("running"):
    L.append(f"   {B}IN FLIGHT{R}")
    for r in sorted(d["running"], key=lambda x:(-x["epochs"],x["set"])):
        ep=r["epochs"]; blk="▓"*min(ep,20)
        L.append(f"     {r['model'][:16]:<16s} {r['set']}/{r['yr']} s{r['seed']:<2d} {C}{blk}{R} {ep} ep")
if d.get("recent"):
    L.append("")
    L.append(f"   {B}RECENTLY COMPLETED{R}")
    for r in d["recent"][:6]:
        L.append(f"     {D}{r['model'][:16]:<16s} {r['set']}/{r['yr']} s{r['seed']:<2d}  {r['epochs']:>2d} ep  {r['min']:>5.1f} min{R}")
# ---- scored cells: ensemble IC and Algorithm 2 trading return, read from score_cells.py's cache.
# BOTH are shown because they dissociate. Algorithm 2 only opens a position on days when all top-L
# predictions are >0 and all bottom-S are <0, so a model whose predictions barely move can keep a
# healthy-looking IC while never trading -- that is exactly how Crossformer failed here. `disp` and
# `gate` are the fields that tell an honest signal from a collapsed one, so they stay on screen
# next to the headline numbers rather than being summarised away.
cells=[c for c in d.get("cells",[]) if c.get("ic") is not None]
active={a["model"] for a in d["arms"] if 0 < a["done"] < a["total"]}
show=[c for c in cells if c["model"] in active] or cells
if show:
    L.append("")
    L.append(f"   {B}SCORED CELLS{R} {D}(seed-mean ensemble · IC + Algorithm 2 return){R}")
    L.append(f"     {D}{'model':<14s}{'cell':>11s} {'n':>2s}  {'IC':>8s} {'t':>6s}  {'ret%':>8s}  {'disp':>9s}  {'gate':>9s}{R}")
    for c in sorted(show,key=lambda x:-(x['ic'] or 0))[:14]:
        ic=c["ic"]; icc = G if ic>0.010 else (Y if ic>0 else RD)
        rt=c.get("ret")
        rts = f"{rtc(rt)}{rt:+8.2f}{R}" if rt is not None else f"{D}{'-':>8s}{R}"
        flag = f" {Y}COLLAPSED{R}" if (c.get("sd") or 1)<1e-4 else ""
        tv=c.get("t"); ts=f"{tv:+6.2f}" if tv is not None else "     -"
        L.append(f"     {c['model'][:14]:<14s}{c['set']+'/'+str(c['yr']):>11s} {c['n']:>2d}  "
                 f"{icc}{ic:+8.4f}{R} {D}{ts}{R}  {rts}  {c['sd']:9.6f}  "
                 f"{c['gate']:>4d}/{c['days']:<4d}{flag}")
    rets=[c["ret"] for c in show if c.get("ret") is not None]
    L.append(f"     {D}{sum(1 for c in show if c['ic']>0)}/{len(show)} cells IC>0 · "
             f"mean IC {sum(c['ic'] for c in show)/len(show):+.4f}"
             + (f" · mean ret {sum(rets)/len(rets):+.2f}% ({sum(1 for r in rets if r>0)}/{len(rets)} up)" if rets else "")
             + f"{R}")
print("\n".join(L))
PY
)
    printf '\033[H\033[2J%s\n' "$frame"
  else
    printf '\033[s\033[1;1H\033[33m  [poll failed %s — keeping last frame]\033[0m\033[u' "$(date +%H:%M:%S)"
  fi
  sleep "$REFRESH"
done
