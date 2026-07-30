#!/bin/bash
# Is the Anvil GPU allocation live yet? Run this on an Anvil LOGIN node.
#
#   bash scripts/transformer_bench/check_gpu_ready.sh
#
# Checks, in the order that things actually fail:
#   1. does a -gpu allocation exist and have hours
#   2. is this user associated with it in Slurm
#   3. is the gpu partition up and are nodes idle
#   4. (optional, with --submit) a ~1 minute gpu-debug job that proves torch sees a GPU
#
# GPU hours on Anvil bill to a SEPARATE allocation from CPU, named with a -gpu suffix. Our CPU
# account is cis251123, so the GPU one is cis251123-gpu. Until that line appears in `mybalance`,
# no GPU job will queue and the rejection message is a cryptic account error.
set -uo pipefail

CPU_ACCT=${CPU_ACCT:-cis251123}
GPU_ACCT=${GPU_ACCT:-${CPU_ACCT}-gpu}
SUBMIT=0
[ "${1:-}" = "--submit" ] && SUBMIT=1

hr() { printf '%s\n' "------------------------------------------------------------"; }
ready=1

hr; echo "1. ALLOCATIONS  (mybalance)"; hr
if command -v mybalance >/dev/null 2>&1; then
  mybalance
  if mybalance 2>/dev/null | grep -q -- "$GPU_ACCT"; then
    echo "   -> '$GPU_ACCT' FOUND"
  else
    echo "   -> '$GPU_ACCT' NOT PRESENT. The allocation is not active yet; nothing else will work."
    ready=0
  fi
else
  echo "   mybalance not on PATH (are you on an Anvil login node?)"; ready=0
fi

hr; echo "2. SLURM ASSOCIATION  (sacctmgr)"; hr
sacctmgr -n show assoc user="$USER" format=account%20,partition%15,qos%20 2>/dev/null \
  | sed 's/^/   /' || echo "   sacctmgr unavailable"
if sacctmgr -n show assoc user="$USER" format=account 2>/dev/null | grep -q -- "$GPU_ACCT"; then
  echo "   -> associated with $GPU_ACCT"
else
  echo "   -> NOT associated with $GPU_ACCT yet"; ready=0
fi

hr; echo "3. GPU PARTITIONS  (sinfo)"; hr
sinfo -p gpu,gpu-debug -o "   %.12P %.6a %.10l %.6D %.6t %N" 2>/dev/null || echo "   sinfo failed"
# GPU nodes are SHARED, so they sit in 'mix' whenever any GPU on them is in use. Counting 'idle'
# nodes is the wrong metric here -- it is normally zero even when GPUs are free. Count nodes that
# are usable (not down/drain) instead.
up=$(sinfo -h -p gpu -o "%t %D" 2>/dev/null | grep -Ev 'down|drain|fail' | awk '{s+=$2} END {print s+0}')
echo "   usable gpu nodes (mix/alloc/idle, excluding down+drain): ${up:-?}"

hr; echo "4. QUOTA REMINDERS"; hr
echo "   max 12 GPUs in use per user, 32 per allocation"
echo "   gpu partition: 48 h max walltime   |   gpu-debug: 0.5 h max"

hr
if [ "$ready" = "1" ]; then
  echo "READY. Next steps:"
  echo "   bash scripts/transformer_bench/setup_anvil.sh"
  echo "   sbatch --export=ALL,MODEL=Crossformer,SMOKE=1 --partition=gpu-debug \\"
  echo "          --time=00:25:00 --array=1 scripts/transformer_bench/anvil_transformer.sb"
else
  echo "NOT READY -- see the failing section above. Re-run this script after approval."
fi
hr

if [ "$SUBMIT" = "1" ] && [ "$ready" = "1" ]; then
  echo; echo "Submitting a 1-minute gpu-debug probe..."
  mkdir -p slurm_logs
  cat > /tmp/_gpuprobe.sb <<EOF
#!/bin/bash
#SBATCH --account=$GPU_ACCT
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --time=00:05:00
#SBATCH --job-name=gpuprobe
#SBATCH --output=slurm_logs/gpuprobe_%j.out
module --force purge
module load modtree/gpu
echo "host: \$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [ -x "$PWD/external/tfenv/bin/python" ]; then
  "$PWD/external/tfenv/bin/python" -c "import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
else
  echo "(venv not built yet -- run setup_anvil.sh to test torch)"
fi
EOF
  sbatch /tmp/_gpuprobe.sb
  echo "Watch with:  squeue -u \$USER    then read slurm_logs/gpuprobe_<jobid>.out"
fi
