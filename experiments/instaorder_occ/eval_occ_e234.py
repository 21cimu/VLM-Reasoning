"""Prepare, run and score the InstaOrder-Occ E2–E4 experiments."""
import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from evidence import (ARMS, MARKS, NOTES, digest, get_masks, load_cases, load_coco,
                      load_reviews, model_prompt, read_jsonl, render, review_template,
                      text_inputs)

HERE = Path(__file__).resolve().parent
DATASET = Path("03_datasets/instaorder_occ.jsonl")
COCO = Path("02_dataset_construction/raw_annotations/instances_val2017.json")
EVALUATION = Path("04_evaluation_and_analysis")
MODEL_NAMES = {"treevgr": "TreeVGR-7B", "qwen": "Qwen2.5-VL-7B-Instruct"}
ARM_GROUPS = {"E3": MARKS, "E4": NOTES}
FAMILIES = {
    "E1": (("E0", "E1"),),
    "E2": (("E0", "E2"), ("E1", "E2")),
    "E3": (("E0", MARKS[0]), (MARKS[1], MARKS[0]), (MARKS[2], MARKS[0])),
    "E4": (("E0", NOTES[0]), (NOTES[1], NOTES[0]), (NOTES[2], NOTES[0])),
}
VISUAL = ("E0", "E1", "E2", *MARKS)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_record(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def blind_export(cases, out, seed):
    out.mkdir(parents=True, exist_ok=False)
    folder = out / "images"
    folder.mkdir()
    ordered = sorted(cases, key=lambda case: case.case_id)
    random.Random(seed).shuffle(ordered)
    copied = {}
    with (out / "cases.jsonl").open("x", encoding="utf-8") as index, \
         (out / "reviews.jsonl").open("x", encoding="utf-8") as templates:
        for case in ordered:
            if case.image_path not in copied:
                image, sha = case.load_image()
                # Use the original pixels and remove filename, EXIF and other metadata.
                image.info.clear()
                pic = f"images/{case.case_id}.png"
                image.save(out / pic)
                copied[case.image_path] = pic, sha
            pic, sha = copied[case.image_path]
            write_record(index, {"case_id": case.case_id, "image": pic,
                                 "question": case.row["question"]})
            write_record(templates, review_template(case.case_id, sha))


def prepare(case, anns, review, arms, seed, allow_provisional=False):
    image, sha = case.load_image()
    masks = None
    if "E2" in arms or set(arms).intersection(MARKS):
        masks = get_masks(case, anns, image.size)
    variants, unavailable = render(image, case, masks, arms, seed)
    notes = {}
    if set(arms).intersection(NOTES):
        notes, reason = text_inputs(review, sha, allow_provisional)
        for arm in set(arms).intersection(NOTES):
            if reason:
                unavailable[arm] = reason
            else:
                variants[arm] = image, None
    return sha, variants, notes or {}, unavailable


def audit(cases, anns, reviews, args):
    args.out.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    with (args.out / "cases.jsonl").open("x", encoding="utf-8") as stream:
        for i, case in enumerate(cases):
            sha, variants, _, unavailable = prepare(
                case, anns, reviews.get(case.case_id), ARMS, args.seed, args.allow_provisional_reviews)
            counts.update(variants.keys())
            counts.update(unavailable.values())
            write_record(stream, {"id": case.row["id"], "case_id": case.case_id,
                                  "image_sha256": sha, "available": list(variants),
                                  "unavailable": unavailable,
                                  "markers": {arm: variants[arm][1] for arm in MARKS if arm in variants}})
            if i < args.preview:
                folder = args.out / "previews" / case.case_id
                folder.mkdir(parents=True)
                for arm in VISUAL:
                    if arm in variants:
                        variants[arm][0].save(folder / f"{arm}.png")
    stats = {"cases": len(cases), "seed": args.seed,
             "e4_review_mode": "provisional" if args.allow_provisional_reviews else "human_reviewed",
             "available": {arm: counts[arm] for arm in ARMS},
             "unavailable_cases": {reason: counts[reason] // 3 for reason in
                                   ("no_boundary", "image_too_small", "no_unrelated_control",
                                    "no_random_control", "boundary_too_close_to_corner",
                                    "review_pending", "review_excluded")}}
    dump(args.out / "summary.json", stats)
    print(json.dumps(stats, ensure_ascii=False))


def import_runner(study):
    sys.path.insert(0, str(study / EVALUATION))
    from eval_blue2_pilot import PilotRunner, extract_answer_letter, extract_boxes
    return PilotRunner, extract_answer_letter, extract_boxes


def run_protocol(args):
    root = args.model_root or Path(os.environ.get("VLM_MODEL_ROOT", "/home/ubuntu/projects/vlm/VLM-reasoning/models"))
    checkpoint = (root / MODEL_NAMES[args.model]).resolve()
    if not (checkpoint / "config.json").is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint}; set --model-root")
    files = sorted(checkpoint.glob("*.safetensors")) or sorted(checkpoint.glob("pytorch_model*.bin"))
    if not files:
        raise ValueError(f"No local model weights found in {checkpoint}")
    weight_files = []
    for file in files:
        info = file.stat()
        weight_files.append([file.name, info.st_size, info.st_mtime_ns])
    sources = [Path(__file__), HERE / "evidence.py", args.study / EVALUATION / "eval_blue2_pilot.py"]
    configs = {p.name: digest(p.read_bytes()) for p in sorted(checkpoint.glob("*.json"))}
    # Weight filenames, sizes and timestamps identify the local snapshot without hashing gigabytes.
    return {"model": args.model, "checkpoint": str(checkpoint), "weight_files": weight_files, "configs": configs,
            "dependencies": {name: version(name) for name in
                             ("torch", "transformers", "qwen-vl-utils", "Pillow", "numpy", "pycocotools")},
            "code": {p.name: digest(p.read_bytes()) for p in sources},
            "dataset_sha256": digest((args.study / DATASET).read_bytes()),
            "coco_sha256": digest((args.study / COCO).read_bytes()), "seed": args.seed}


@contextlib.contextmanager
def run_lock(path):
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"{lock} exists; confirm no process is using the run before removing a stale lock") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"{os.getpid()}\n")
        yield
    finally:
        lock.unlink()


def resume(path):
    if not path.exists():
        return {}
    # A process can be interrupted between writing a JSON object and its newline.
    with path.open("rb+") as stream:
        data = stream.read()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            stream.truncate(cut)
    return index_predictions([path])


def run(cases, anns, reviews, args):
    arms = tuple(dict.fromkeys(arm for group in args.arms for arm in ARM_GROUPS.get(group, (group,))))
    e4_mode = None
    if set(arms).intersection(NOTES):
        if args.reviews is None:
            raise ValueError("E4 requires --reviews")
        e4_mode = "provisional" if args.allow_provisional_reviews else "human_reviewed"
    elif args.allow_provisional_reviews:
        raise ValueError("--allow-provisional-reviews requires an E4 condition")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    protocol = run_protocol(args)
    signature = digest(json.dumps(protocol, sort_keys=True).encode())
    review_sha = digest(args.reviews.read_bytes()) if args.reviews else None
    metadata = {"protocol": protocol, "protocol_sha256": signature, "review_sha256": review_sha,
                "e4_review_mode": e4_mode}
    manifest = args.out.with_suffix(args.out.suffix + ".meta.json")
    with run_lock(args.out):
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Run inputs or reviewed notes changed; use a fresh --out")
        elif args.out.exists():
            raise ValueError("Prediction JSONL exists without its metadata file; use a fresh --out")
        else:
            dump(manifest, metadata)
        finished = resume(args.out)
        selected = {case.row["id"]: case for case in cases}
        for row in finished.values():
            if row.get("protocol_sha256") != signature or row.get("model") != args.model:
                raise ValueError("Existing prediction belongs to a different run")
        runner = None
        with args.out.open("a", encoding="utf-8", newline="\n") as stream:
            for i, case in enumerate(selected.values(), 1):
                row = case.row
                pending = [arm for arm in arms if (row["id"], arm) not in finished]
                if pending:
                    sha, variants, notes, unavailable = prepare(
                        case, anns, reviews.get(case.case_id), pending, args.seed, args.allow_provisional_reviews)
                else:
                    sha = digest(case.image_path.read_bytes())
                if any(record["image_sha256"] != sha
                       for arm in ARMS if (record := finished.get((row["id"], arm)))):
                    raise ValueError(f"{row['id']}: source image changed since the recorded results")
                for arm in pending:
                    record = {"id": row["id"], "case_id": case.case_id, "image_id": row["image_id"],
                              "stratum": row["stratum"], "gt": row["gt"], "condition": arm,
                              "model": args.model, "protocol_sha256": signature, "image_sha256": sha}
                    if arm in NOTES:
                        record.update(review_sha256=review_sha, e4_review_mode=e4_mode)
                    if arm in unavailable:
                        record.update(status="unavailable", reason=unavailable[arm])
                    else:
                        if runner is None:
                            runner_cls, extract_letter, extract_boxes = import_runner(args.study)
                            runner = runner_cls(args.model, model_root=str(Path(protocol["checkpoint"]).parent))
                        image, marker = variants[arm]
                        prompt = model_prompt(row["question"], row["options"], arm, notes.get(arm))
                        started = time.perf_counter()
                        raw, proc_wh = runner.infer([image], prompt)
                        letter = extract_letter(raw, set(row["valid_opts"]))
                        record.update(status="ok", pred_letter=letter, correct=int(letter == row["gt"]),
                                      pred_boxes=extract_boxes(raw), proc_wh=proc_wh, raw=raw, prompt=prompt,
                                      marker=marker, elapsed_sec=round(time.perf_counter() - started, 4))
                    write_record(stream, record)
                if i % 20 == 0 or i == len(selected):
                    print(f"{args.model}: {i}/{len(selected)}", flush=True)


def index_predictions(paths):
    records = {}
    for path in paths:
        for rec in read_jsonl(path):
            key = rec["id"], rec["condition"]
            if key in records or rec["condition"] not in ARMS or rec["status"] not in ("ok", "unavailable"):
                raise ValueError(f"{path}: duplicate or invalid prediction {key}")
            records[key] = rec
    return records


def historical_cohorts(paths, study, cases):
    sys.path.insert(0, str(study / EVALUATION / "analysis"))
    from analyze_blue2_pilot import matched_1to1_iou05

    by_id = {case.row["id"]: case.row for case in cases}
    observed, correct, wrong = set(), set(), set()
    for rec in (record for path in paths for record in read_jsonl(path)):
        if rec["condition"] != "A_full":
            continue
        case_id = rec["id"]
        row = by_id.get(case_id)
        if row is None or case_id in observed:
            raise ValueError(f"Historical E0 contains unknown or duplicate {case_id}")
        if rec["dataset"] != "InstaOrder-Occ" or any(rec[key] != row[key] for key in ("gt", "gt_boxes", "img_size")):
            raise ValueError(f"Historical record is incompatible with InstaOrder-Occ: {case_id}")
        prediction = rec["pred_letter"]
        if prediction not in ("", *row["valid_opts"]) or rec["correct"] != int(prediction == row["gt"]):
            raise ValueError(f"Historical correctness is inconsistent: {case_id}")
        observed.add(case_id)
        hit, _ = matched_1to1_iou05(rec["pred_boxes"], row["gt_boxes"], rec.get("proc_wh"), row["img_size"])
        if hit:
            (correct if rec["correct"] else wrong).add(case_id)
    if not observed:
        raise ValueError("Historical E0 has no A_full records")
    return {"historical_treevgr_both_boxes": correct | wrong,
            "historical_treevgr_both_boxes_correct": correct,
            "historical_treevgr_both_boxes_wrong": wrong}, len(observed)


def paired(ids, base, target, records, image_ids):
    cells, shifts_by_image = Counter(), Counter()
    invalid_base = invalid_target = 0
    for case_id in ids:
        first = records[case_id, base]
        second = records[case_id, target]
        a, b = first["correct"], second["correct"]
        cells[a, b] += 1
        shifts_by_image[image_ids[case_id]] += b - a
        invalid_base += first["pred_letter"] == ""
        invalid_target += second["pred_letter"] == ""
    n = len(ids)
    gain, loss = cells[0, 1], cells[1, 0]
    wrong, correct = cells[0, 0] + gain, cells[1, 1] + loss
    discordant = gain + loss
    p_value = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(gain, loss) + 1)) / (2 ** discordant))

    # Pairwise exchanges can be correlated within a COCO image. Flip all its pairs together.
    signed_sums = Counter({0: 1})
    for shift in shifts_by_image.values():
        if shift:
            next_sums = Counter({key + shift: count for key, count in signed_sums.items()})
            next_sums.update({key - shift: count for key, count in signed_sums.items()})
            signed_sums = next_sums
    cluster_p = sum(count for key, count in signed_sums.items() if abs(key) >= abs(gain - loss)) / sum(signed_sums.values())
    return {"base": base, "target": target, "n_pairs": n, "n_images": len(shifts_by_image),
            "both_wrong": cells[0, 0], "wrong_to_right": gain, "right_to_wrong": loss,
            "both_right": cells[1, 1], "base_invalid": invalid_base, "target_invalid": invalid_target,
            "base_accuracy": correct / n if n else None,
            "target_accuracy": (gain + cells[1, 1]) / n if n else None,
            "delta_pp": 100 * (gain - loss) / n if n else None,
            "correction_rate": gain / wrong if wrong else None,
            "harm_rate": loss / correct if correct else None,
            "mcnemar_exact_p": p_value if n else None,
            "image_cluster_exact_p": cluster_p if n else None}


def score(cases, args):
    records = index_predictions(args.predictions)
    if not records:
        raise ValueError("No prediction rows")
    by_id = {case.row["id"]: case for case in cases}
    image_ids = {key: case.row["image_id"] for key, case in by_id.items()}
    signatures, review_sources = set(), set()
    image_hashes = {}
    for (key, arm), rec in records.items():
        if key not in by_id:
            raise ValueError(f"Unknown case {key}")
        source = by_id[key]
        expected = {"case_id": source.case_id, "gt": source.row["gt"], "image_id": image_ids[key],
                    "stratum": source.row["stratum"], "model": args.model}
        if any(rec.get(field) != value for field, value in expected.items()):
            raise ValueError(f"Dataset or model mismatch for {key}/{arm}")
        signatures.add(rec["protocol_sha256"])
        if arm in NOTES and rec["status"] == "ok":
            review_mode = rec.get("e4_review_mode", "human_reviewed")
            if review_mode not in ("human_reviewed", "provisional"):
                raise ValueError(f"Invalid E4 review mode for {key}/{arm}")
            review_sources.add((rec["review_sha256"], review_mode))
        sha = rec["image_sha256"]
        if image_hashes.setdefault(key, sha) != sha:
            raise ValueError(f"Source image mismatch across arms for {key}")
        if rec["status"] == "ok":
            pred = rec["pred_letter"]
            if pred not in ("", "A", "B") or rec["correct"] != int(pred == source.row["gt"]):
                raise ValueError(f"Invalid correctness record {key}/{arm}")
    if len(signatures) != 1:
        raise ValueError("Prediction files must share the same model, prompt protocol, data and seed")
    if len(review_sources) > 1:
        raise ValueError("Scored E4 arms must share the same review file and review mode")
    review_sha, review_mode = next(iter(review_sources), (None, None))
    cohorts = {"all": set(by_id)}
    for stratum in sorted({case.row["stratum"] for case in cases}):
        cohorts[stratum] = {key for key, case in by_id.items() if case.row["stratum"] == stratum}
    historical = None
    if args.treevgr_e0:
        extra, observed = historical_cohorts(args.treevgr_e0, args.study, cases)
        cohorts.update(extra)
        counts = [len(extra[key]) for key in extra]
        historical = {"sources": [str(path) for path in args.treevgr_e0], "n_A_full": observed,
                      "both_boxes": counts[0], "correct": counts[1], "wrong": counts[2],
                      "matches_documented_580_400_238_162": [observed, *counts] == [580, 400, 238, 162]}
    report = {"model": args.model, "protocol_sha256": next(iter(signatures)),
              "prediction_files": [str(path) for path in args.predictions],
              "review_sha256": review_sha, "e4_review_mode": review_mode,
              "historical_selection": historical, "cohorts": {}}
    for label, ids in cohorts.items():
        arm_counts = {}
        for arm in ARMS:
            observed = [records[key, arm] for key in ids if (key, arm) in records]
            ok = [rec for rec in observed if rec["status"] == "ok"]
            correct = sum(rec["correct"] for rec in ok)
            arm_counts[arm] = {"scored": len(ok), "correct": correct,
                               "accuracy": correct / len(ok) if ok else None,
                               "missing": len(ids) - len(observed),
                               "unavailable": dict(Counter(rec["reason"] for rec in observed if rec["status"] == "unavailable")),
                               "invalid_answer": sum(not rec["pred_letter"] for rec in ok)}
        family_results = {}
        for family, contrasts in FAMILIES.items():
            needed = set(sum(contrasts, ()))
            common = sorted(key for key in ids if all(records.get((key, arm), {}).get("status") == "ok" for arm in needed))
            family_results[family] = {"required_arms": sorted(needed), "common_n": len(common),
                                      "comparisons": [paired(common, a, b, records, image_ids) for a, b in contrasts]}
        report["cohorts"][label] = {"total": len(ids), "available_by_arm": arm_counts, "families": family_results}
    dump(args.out, report)
    print(f"{args.model}: report written to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=HERE.parents[1], help="Existing stage-1 package root")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "annotate", "run", "score"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, type=Path)
        if name != "score":
            p.add_argument("--limit", type=int, default=0, help="First N rows, 0 means all")
            p.add_argument("--seed", type=int, default=42)
        if name in ("audit", "run"):
            p.add_argument("--reviews", type=Path)
            p.add_argument("--allow-provisional-reviews", action="store_true",
                           help="Use complete pending texts for an exploratory E4 run")
        if name in ("run", "score"):
            p.add_argument("--model", required=True, choices=MODEL_NAMES)
        if name == "audit":
            p.add_argument("--preview", type=int, default=0)
        if name == "run":
            p.add_argument("--arms", nargs="+", choices=(*ARMS, *ARM_GROUPS),
                           default=["E0", "E1", "E2", "E3"])
            p.add_argument("--model-root", type=Path)
        if name == "score":
            p.add_argument("--predictions", nargs="+", type=Path, required=True)
            p.add_argument("--treevgr-e0", nargs="+", type=Path,
                           help="Historical TreeVGR pilot prediction JSONL or its GPU shards")
    args = parser.parse_args()
    args.study = args.study.resolve()
    if getattr(args, "limit", 0) < 0 or getattr(args, "preview", 0) < 0:
        parser.error("--limit and --preview must be nonnegative")
    if getattr(args, "allow_provisional_reviews", False) and args.reviews is None:
        parser.error("--allow-provisional-reviews requires --reviews")
    try:
        all_cases = load_cases(args.study / DATASET)
        cases = all_cases[:args.limit] if getattr(args, "limit", 0) else all_cases
        if args.command == "annotate":
            blind_export(cases, args.out, args.seed)
        elif args.command == "score":
            score(cases, args)
        else:
            anns = load_coco(args.study / COCO, cases)
            reviews = load_reviews(args.reviews, all_cases)
            if args.command == "audit":
                audit(cases, anns, reviews, args)
            else:
                run(cases, anns, reviews, args)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
