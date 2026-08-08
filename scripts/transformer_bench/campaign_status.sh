#!/bin/bash
# One-shot campaign status from the pod, as JSON on stdout. Used by monitor_campaign.sh for the
# live view and by the status artifact; kept separate so both read exactly the same numbers.
#
#   bash scripts/transformer_bench/campaign_status.sh USER@HOST:PORT [REMOTE_REPO]
#
# Everything is derived on the pod in ONE ssh round trip -- the link has been flaky all campaign,
# and a monitor that opens six connections per refresh fails far more often than one that opens one.
set -uo pipefail
HOST=${1:?usage: campaign_status.sh USER@HOST[:PORT] [REMOTE_REPO]}
REMOTE=${2:-/workspace/EXAMM-Extended}
PORT=22; case "$HOST" in *:*) PORT="${HOST##*:}"; HOST="${HOST%:*}" ;; esac
KEY=${SSH_KEY:-$HOME/.ssh/runpod_examm}

ssh -p "$PORT" -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "cd $REMOTE 2>/dev/null || exit 7
PY=\$(cat external/.tfenv_path)/bin/python
\$PY - <<'PYEOF'
import json, os, glob, subprocess, re
ROOT='results/transformer_bench/mid_highmid'
ARMS=[('PatchTSTOfficial','PatchTSTOfficial'),('DeformTime','DeformTime'),
      ('DLinearOfficial','DLinearOfficial'),('ITransformerOfficial','iTransformer'),
      ('CrossformerOfficial','Crossformer')]
out={'arms':[],'running':[],'gpu':[],'recent':[]}
for key,label in ARMS:
    done=len([f for f in glob.glob(f'{ROOT}/*/author/**/{key}/L96/seed_*/.done',recursive=True)])
    if done or key in ('PatchTSTOfficial','DeformTime','DLinearOfficial'):
        out['arms'].append({'model':label,'done':done,'total':120})
# in-flight: parse ps for model_id, then count epochs in that run's train.log
ps=subprocess.run(['ps','-eo','args','--no-headers'],capture_output=True,text=True).stdout
ids=sorted(set(re.findall(r'--model_id (\S+)',ps)))
for mid in ids:
    m=re.match(r'(\w+)_author_(?:(\w+?)_)?L96_(set\d)_cohort_(\d{4})_aligned_seed_(\d+)',mid)
    if not m: continue
    model,var,st,coh,seed=m.groups()
    var=var or ''
    d=f\"{ROOT}/{st}/author/{'var_'+var+'/' if var else ''}cohort_{coh}_aligned/{model}/L96/seed_{seed}\"
    ep=0
    try: ep=open(d+'/train.log',errors='ignore').read().count('cost time')
    except OSError: pass
    out['running'].append({'model':model,'set':st,'yr':int(coh)+2,'seed':int(seed),'epochs':ep})
try:
    g=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
    out['gpu']=[{'i':int(a),'util':int(b),'mem':int(c)} for a,b,c in
                (l.split(', ') for l in g.strip().splitlines() if l.strip())]
except Exception: pass
# 8 most recent completions, with their timing
tj=[]
for f in glob.glob(f'{ROOT}/**/timing.json',recursive=True):
    d=os.path.dirname(f)
    if not os.path.exists(d+'/.done'): continue
    tj.append((os.path.getmtime(f),f))
for _,f in sorted(tj)[-8:]:
    try:
        t=json.load(open(f))
        out['recent'].append({'model':t['model'],'set':t.get('set'),'yr':int(t['cohort'][7:11])+2,
                              'seed':t['seed'],'epochs':t['epochs'],'min':round(t['wall_sec']/60,1)})
    except Exception: pass
out['recent'].reverse()
# ---- per-cell ensemble IC + Algorithm 2 trading return.
# READ FROM THE CACHE, NEVER COMPUTED HERE. Trading shells out to trade_portfolio.py per cell over
# 50 tickers, which on a box already running nine trainings takes far longer than the monitor's
# refresh interval -- computing inline would mean every poll stalls or times out. score_cells.py
# --loop owns the computing and keeps .cell_scores.json warm; this just reads it, so a poll is a
# file read regardless of how many cells exist. If the daemon is not running the numbers simply go
# stale rather than disappearing, which is the right failure: a stale IC is still informative,
# a hung monitor is not.
cells={}
try:
    for rec in json.load(open('results/transformer_bench/.cell_scores.json')).values():
        k=(rec['model'],rec.get('set'),rec['yr'])
        # keyed on the seed list, so one cell can hold several entries as seeds accumulate; keep
        # the widest ensemble, which is always the most recent.
        if k not in cells or rec['n']>cells[k]['n']: cells[k]=rec
except Exception: pass
out['cells']=sorted(cells.values(),key=lambda x:(x['model'],x['set'] or '',x['yr']))
out['shards']=len(re.findall(r'run_portfolio_campaign',ps))
print(json.dumps(out))
PYEOF"
