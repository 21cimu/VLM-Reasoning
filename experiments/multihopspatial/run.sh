#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "$mode" in
  smoke) count=5 ;;
  prepare|full) count=4500 ;;
  --help) printf 'Usage: bash run.sh {prepare|smoke|full}\n'; exit 0 ;;
  *) printf 'Usage: bash run.sh {prepare|smoke|full}\n' >&2; exit 2 ;;
esac

experiment="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${VLM_ROOT_DIR:-/home/ubuntu/vLM-reasoning}"
run_dir="${VLM_RUN_DIR:-$root/runs/multihopspatial}"
mkdir -p "$run_dir/results"
cd "$run_dir"

exec 9>.run.lock
flock -n 9 || { printf 'Another MultihopSpatial run is using %s\n' "$PWD" >&2; exit 1; }

author_commit=b1b6a7be261e3cde814e117eabe330d5ef47c1f0
model_revision=c7148d4d58d4708032b5386ade8bb1b8ac6219a5
data_revision=6c962f5227acc133d2196cbf89f09441c7ccb703
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ ! -x .venv/bin/python ]]; then
  "${PYTHON_BIN:-python3}" -m venv .venv
fi
python="$PWD/.venv/bin/python"
"$python" -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
"$python" -m pip install -r "$experiment/requirements-eval.txt"
if [[ "$mode" != prepare ]]; then
  "$python" "$experiment/parallel_eval.py" --check-devices
fi

if [[ ! -e author-code/.git ]]; then
  git clone --no-checkout https://github.com/youngwanLEE/multihopspatial.git author-code
fi
if ! git -C author-code cat-file -e "$author_commit^{commit}"; then
  git -C author-code fetch origin "$author_commit"
fi
if [[ -f author-code/.git/index ]] && ! git -C author-code diff --quiet HEAD --; then
  printf 'Tracked files have local changes in %s/author-code\n' "$PWD" >&2
  exit 1
fi
git -C author-code checkout --detach "$author_commit"

"$python" - "$model_revision" "$data_revision" "$count" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

model_revision, data_revision, count = sys.argv[1:]
model_dir = Path("model-4b")
model_id = "etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct"
model_files = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "chat_template.jinja", "tokenizer.json",
    "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "model.safetensors.index.json",
    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
)
missing_weights = [name for name in model_files if not (model_dir / name).is_file()]
if missing_weights:
    snapshot_download(model_id, revision=model_revision, local_dir=model_dir,
                      allow_patterns=missing_weights, max_workers=4)

data_id = "etri-vilab/MultihopSpatial"
data_dir = Path("benchmark")
label_path = "data/multihop_test_4500.json"
labels = data_dir / label_path
if not labels.is_file():
    hf_hub_download(data_id, label_path, repo_type="dataset",
                    revision=data_revision, local_dir=data_dir)

rows = json.loads(labels.read_bytes())
if len(rows) != 4500:
    raise SystemExit(f"Expected 4,500 official questions, found {len(rows)}.")
images = {f"data/images/{row['image_path'].lstrip('/')}" for row in rows[:int(count)]}
missing_images = sorted(name for name in images if not (data_dir / name).is_file())
if missing_images:
    snapshot_download(data_id, repo_type="dataset", revision=data_revision,
                      local_dir=data_dir, allow_patterns=missing_images, max_workers=4)
missing_images = sorted(name for name in images if not (data_dir / name).is_file())
if missing_images:
    raise SystemExit(f"Missing {len(missing_images)} test images; first: {missing_images[0]}")
print(f"Assets present: {len(images)} images for {count} questions.", flush=True)
PY
sha256sum -c "$experiment/assets.sha256"

if [[ "$mode" == prepare ]]; then
  printf 'MultihopSpatial assets verified at %s. After attaching a GPU, run smoke.\n' "$PWD"
  exit 0
fi

output_dir="$(mktemp -d "$PWD/results/$mode-$(TZ=Asia/Hong_Kong date +%Y%m%d-%H%M%S)-XXXXXX")"
printf 'Result directory: %s\n' "$output_dir"
{
  printf 'experiment_code\t%s\n' "$(git -C "$experiment" rev-parse HEAD)"
  printf 'author_code\t%s\nmodel_revision\t%s\ndata_revision\t%s\n' \
    "$author_commit" "$model_revision" "$data_revision"
  printf 'backend\ttransformers greedy\nmode\t%s\nexpected_rows\t%s\n' "$mode" "$count"
} > "$output_dir/versions.tsv"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$python" -u "$experiment/parallel_eval.py" \
  "$count" "$output_dir" 2>&1 | tee "$output_dir/console.log"
