"""Run the pinned author's evaluator on independent single-GPU shards."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

AUTHOR_EVAL = Path("author-code/eval").resolve()
TEST_JSON = Path("benchmark/data/multihop_test_4500.json")


def write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def available_devices():
    import torch

    visible = torch.cuda.device_count()
    if visible == 0:
        raise SystemExit("CUDA sees 0 devices. Use prepare to stage assets, then attach a GPU.")
    try:
        requested = int(os.environ.get("VLM_GPU_COUNT", str(min(4, visible))))
    except ValueError:
        raise SystemExit("VLM_GPU_COUNT must be a positive integer.") from None
    if not 1 <= requested <= visible:
        raise SystemExit(f"Requested {requested} GPUs, but CUDA sees {visible}. Set VLM_GPU_COUNT to an available count.")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = mask.split(",") if mask is not None else [str(i) for i in range(visible)]
    return [device.strip() for device in devices[:requested]]


def author_evaluator():
    sys.path.insert(0, str(AUTHOR_EVAL))
    import benchmark_qwen

    return benchmark_qwen


def evaluate_shard(path):
    stats = author_evaluator().run_benchmark(
        json_path=str(path / "questions.json"),
        image_root="benchmark/data/images",
        output_path=str(path / "predictions.json"),
        model_path="model-4b",
        use_flash_attention=False,
    )
    if stats["errors"]:
        raise SystemExit(f"Author evaluator reported {stats['errors']} errors in {path}.")


def wait_for_workers(jobs):
    pending = dict(jobs)
    while pending:
        for path, process in list(pending.items()):
            status = process.poll()
            if status is None:
                continue
            if status:
                raise SystemExit(f"{path.name} exited with code {status}. See {path / 'console.log'}.")
            del pending[path]
        if pending:
            time.sleep(1)


def run_parallel(questions, devices, output_dir):
    jobs = []
    shard_inputs = []
    try:
        for rank, device in enumerate(devices):
            start = rank * len(questions) // len(devices)
            end = (rank + 1) * len(questions) // len(devices)
            shard = questions[start:end]
            path = output_dir / f"shard-{rank:02d}"
            path.mkdir()
            write_json(path / "questions.json", shard)
            shard_inputs.append((path, shard))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
            with (path / "console.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(path)],
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                )
            jobs.append((path, process))
            print(f"GPU {device}: {len(shard)} questions. Log: {path / 'console.log'}", flush=True)
        wait_for_workers(jobs)
    finally:
        running = [process for _, process in jobs if process.poll() is None]
        for process in running:
            process.terminate()
        for process in running:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    predictions = []
    for path, shard in shard_inputs:
        rows = json.loads((path / "predictions.json").read_bytes())
        if len(rows) != len(shard) or any(row is None or row.get("error") for row in rows):
            raise SystemExit(f"Incomplete results in {path}. See {path / 'console.log'}.")
        if [row["id"] for row in rows] != [row["id"] for row in shard]:
            raise SystemExit(f"Unexpected question IDs or order in {path}.")
        predictions.extend(rows)
    return predictions


def score_predictions(rows):
    mcq = author_evaluator().calculate_mcq_accuracy(rows)
    # Match the author's full-set denominators, threshold and rounding.
    ious = [row["iou"] for row in rows if row.get("score") and row.get("iou") is not None]
    grounded = sum(iou >= 0.5 for iou in ious)
    evaluated = mcq["total_evaluated"]
    return {
        "total": len(rows),
        "mcq_evaluated": evaluated,
        "mcq_correct": mcq["correct"],
        "mcq_accuracy": mcq["accuracy"],
        "acc_at_iou50": round(grounded / evaluated * 100, 2) if evaluated else 0.0,
        "avg_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
    }


def main():
    if sys.argv[1] == "--check-devices":
        print("Selected CUDA devices:", ", ".join(available_devices()))
        return
    if sys.argv[1] == "--worker":
        evaluate_shard(Path(sys.argv[2]))
        return

    count = int(sys.argv[1])
    output_dir = Path(sys.argv[2]).resolve()
    devices = available_devices()
    questions = json.loads(TEST_JSON.read_bytes())[:count]
    if len(questions) != count or count < len(devices):
        raise SystemExit(f"Cannot divide {len(questions)} questions across {len(devices)} GPUs.")
    rows = run_parallel(questions, devices, output_dir)
    write_json(output_dir / "predictions.json", rows)
    metrics = score_predictions(rows)
    metrics["devices"] = devices
    write_json(output_dir / "metrics.json", metrics)

    print(f"MCQ Accuracy: {metrics['mcq_accuracy']}% ({metrics['mcq_correct']}/{metrics['mcq_evaluated']})")
    print(f"Acc@50IoU: {metrics['acc_at_iou50']}%")
    print(f"Avg. IoU (MCQ-correct only): {metrics['avg_iou']}")
    print(f"Saved {len(rows)} rows to {output_dir / 'predictions.json'}.")
    if not metrics["mcq_evaluated"] or not any(row.get("pred_bbox") is not None for row in rows):
        raise SystemExit("The evaluation produced no parsed answers or no parsed boxes.")


if __name__ == "__main__":
    main()
