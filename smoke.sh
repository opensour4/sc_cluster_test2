#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
device=${1:?Usage: bash smoke.sh DEVICE}
if [[ $# != 1 ]]; then echo 'Usage: bash smoke.sh DEVICE' >&2; exit 2; fi
python=${PYTHON:-python}
ranks=${NPROC:-1}
if ! [[ "$ranks" =~ ^[1-9][0-9]*$ ]]; then echo 'NPROC must be a positive integer' >&2; exit 2; fi
data=data/dummy.npz
if [[ -n "${RUN_DIR:-}" ]]; then
  out=$RUN_DIR
  mkdir -p "$(dirname "$out")"
  mkdir "$out" || { echo "Choose a new RUN_DIR for each test." >&2; exit 1; }
else
  mkdir -p runs
  out=$(mktemp -d "runs/cluster-${ranks}-XXXXXX")
fi
echo "Run directory: $out"
common=(--data "$data" --device "$device" --precision "${PRECISION:-bf16}")
if [[ -n "${PLUGIN:-}" ]]; then common+=(--plugin "$PLUGIN"); fi
launcher=("$python")
dist_args=()
if [[ "$ranks" != 1 ]]; then
  launcher+=(-m torch.distributed.run --standalone "--nproc-per-node=$ranks")
  if [[ -n "${BACKEND:-}" ]]; then dist_args+=(--backend "$BACKEND"); fi
fi
# Leave omitted settings to train.py so the direct and shell entry points have identical defaults.
batch_args=()
if [[ -n "${MICRO_GROUPS:-}" ]]; then batch_args+=(--micro-groups "$MICRO_GROUPS"); fi
if [[ -n "${ACCUM:-}" ]]; then batch_args+=(--accum "$ACCUM"); fi
# The conditional expansions also work under macOS Bash 3.2 with nounset.
"${launcher[@]}" train.py "${common[@]}" ${dist_args[@]+"${dist_args[@]}"} \
    --output "$out/checkpoint.pt" --steps "${TRAIN_STEPS:-3}" ${batch_args[@]+"${batch_args[@]}"} \
    --warmup "${WARMUP:-1000}" 2>&1 | tee "$out/train.log"
"$python" infer.py "${common[@]}" --checkpoint "$out/checkpoint.pt" --steps "${INFER_STEPS:-4}" \
    --output "$out/predictions.npz" 2>&1 | tee "$out/infer.log"
"$python" evaluate.py --data "$data" --predictions "$out/predictions.npz" \
    --output "$out/metrics.json" 2>&1 | tee "$out/eval.log"
"$python" check_run.py "$out" --data "$data" 2>&1 | tee "$out/check_run.log"
"$python" verify.py "${common[@]}" 2>&1 | tee "$out/numeric.log"
