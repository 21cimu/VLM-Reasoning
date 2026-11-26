"""Visual interventions and blind review inputs for InstaOrder-Occ."""
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


MARKS = ("E3_boundary", "E3_random", "E3_unrelated")
NOTES = ("E4_verified", "E4_wrong", "E4_neutral")
ARMS = ("E0", "E1", "E2", *MARKS, *NOTES)
NOTE_FIELDS = dict(zip(NOTES, ("verified", "wrong", "neutral")))
DIRECTION = re.compile(
    r"\b(?:option|answer|occlu\w*|cover(?:s|ed|ing)?|in front|frontmost|foreground|"
    r"background|behind|overlap(?:s|ped|ping)?|closer|farther|nearer|obscur\w*|"
    r"block(?:s|ed|ing)?|hidden|hides|occluder)\b|(?:答案|选项|選項|遮挡|遮擋|挡住|擋住|"
    r"前景|背景|在.{0,15}?[前後后][方面边邊]|更靠[前後后]|盖住|蓋住|被挡|被擋)",
    re.IGNORECASE,
)
TEXT_CHECKS = ("image_only", "no_direct_answer", "verified_visible",
               "wrong_contains_false_observation", "neutral_relation_irrelevant")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: JSON object required")
                yield row


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rounded_box(annotation):
    x, y, w, h = annotation["bbox"]
    return [round(v) for v in (x, y, x + w, y + h)]


@dataclass(frozen=True)
class Case:
    row: dict
    image_path: Path
    pair: tuple

    @property
    def case_id(self):
        image_id = self.row["image_id"]
        stable_key = f"occ-e234-v1:{image_id}:{self.pair[0]}:{self.pair[1]}"
        return digest(stable_key.encode())[:24]

    def load_image(self):
        data = self.image_path.read_bytes()
        from io import BytesIO
        with Image.open(BytesIO(data)) as source:
            image = source.convert("RGB")
        if image.size != tuple(self.row["img_size"]):
            raise ValueError(f"{self.row['id']}: image dimensions do not match dataset")
        return image, digest(data)


def load_cases(path, limit=0):
    cases, seen, seen_pairs = [], set(), set()
    for row in read_jsonl(path):
        match = re.fullmatch(r"insta_(\d+)_(\d+)_(\d+)", row["id"])
        if match is None:
            raise ValueError(f"Invalid InstaOrder-Occ id: {row['id']}")
        image_id, first, second = map(int, match.groups())
        # The raw ID orders the true occluder first. The rendering code receives an unordered pair.
        pair = tuple(sorted((first, second)))
        key = (image_id, pair)
        options = row["options"].splitlines()
        if (row["id"] in seen or key in seen_pairs or first == second
                or row["image_id"] != image_id or row["gt"] not in ("A", "B")
                or row["valid_opts"] != ["A", "B"] or len(row["gt_boxes"]) != 2
                or len(options) != 2 or not options[0].startswith("A. ")
                or not options[1].startswith("B. ")):
            raise ValueError(f"Duplicate or invalid case: {row['id']}")
        seen.add(row["id"])
        seen_pairs.add(key)
        cases.append(Case(row, Path(path).parent / row["image_path"], pair))
    if not cases:
        raise ValueError(f"No cases in {path}")
    return cases[:limit] if limit else cases


def load_coco(path, cases):
    wanted = {ann_id for case in cases for ann_id in case.pair}
    with Path(path).open(encoding="utf-8-sig") as stream:
        return {ann["id"]: ann for ann in json.load(stream)["annotations"] if ann["id"] in wanted}


def decode_mask(segmentation, size):
    from pycocotools import mask as coco_mask

    width, height = size
    if isinstance(segmentation, list):
        if not segmentation:
            raise ValueError("Empty polygon segmentation")
        rle = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    else:
        if segmentation["size"] != [height, width]:
            raise ValueError("RLE dimensions do not match image")
        counts = segmentation["counts"]
        if isinstance(counts, list):
            if any(v < 0 for v in counts) or sum(counts) != width * height:
                raise ValueError("Invalid uncompressed RLE length")
            rle = coco_mask.frPyObjects(segmentation, height, width)
        else:
            rle = {"size": [height, width], "counts": counts.encode("ascii")}
    mask = coco_mask.decode(rle).astype(bool)
    if mask.shape != (height, width) or not mask.any():
        raise ValueError("Empty or incorrectly sized instance mask")
    return mask


def get_masks(case, annotations, size):
    rows = [annotations[ann_id] for ann_id in case.pair]
    if any(ann["image_id"] != case.row["image_id"] or ann.get("iscrowd") for ann in rows):
        raise ValueError(f"{case.row['id']}: invalid COCO instance assignment")
    boxes = [rounded_box(ann) for ann in rows]
    box_a, box_b = case.row["gt_boxes"]
    # Verify that both masks belong to the two question targets without consulting the answer.
    if boxes[0] == boxes[1] or sorted(boxes) != sorted((box_a, box_b)):
        raise ValueError(f"{case.row['id']}: ambiguous or inconsistent COCO boxes")
    ordered = [rows[boxes.index(box_a)], rows[boxes.index(box_b)]]
    return tuple(decode_mask(ann["segmentation"], size) for ann in ordered)


def tint(image, mask, color, opacity):
    alpha = Image.fromarray(mask.astype(np.uint8) * opacity)
    return Image.composite(Image.new("RGB", image.size, color), image, alpha)


def dilate(mask, radius):
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0


def free_centers(forbidden, radius):
    """Return centers whose complete marker square fits and avoids forbidden pixels."""
    width = 2 * radius + 1
    integral = np.pad(forbidden, ((1, 0), (1, 0))).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    sums = integral[width:, width:] - integral[:-width, width:] - integral[width:, :-width] + integral[:-width, :-width]
    return np.argwhere(sums == 0) + radius


def mark_centers(first, second, seed):
    height, width = first.shape
    # Same symmetric pair geometry as the existing 15-pixel seam, with no directional order.
    seam = dilate(first, 7) & dilate(second, 7)
    points = np.argwhere(seam)
    if not len(points):
        return None, "no_boundary"
    radius = max(3, round(0.025 * min(height, width)))
    if min(height, width) < 2 * radius + 1:
        return None, "image_too_small"
    # Keep the entire disk inside the image and center it on the seam where possible.
    median = np.median(points, axis=0)
    fitted = np.clip(points, radius, [height - radius - 1, width - radius - 1])
    distance_sq = ((points - fitted) ** 2).sum(axis=1)
    candidates = fitted[distance_sq == 0]
    if not len(candidates):
        candidates = fitted[distance_sq <= radius ** 2]
    if not len(candidates):
        return None, "boundary_too_close_to_corner"
    y, x = candidates[np.argmin(((candidates - median) ** 2).sum(axis=1))]
    forbidden = seam.copy()
    forbidden[y-radius:y+radius+1, x-radius:x+radius+1] = True

    unrelated = free_centers(forbidden | dilate(first | second, 7), radius)
    if not len(unrelated):
        return None, "no_unrelated_control"
    rng = random.Random(seed)
    other_y, other_x = unrelated[rng.randrange(len(unrelated))]
    forbidden[other_y-radius:other_y+radius+1, other_x-radius:other_x+radius+1] = True
    random_centers = free_centers(forbidden, radius)
    if not len(random_centers):
        return None, "no_random_control"
    rand = random_centers[rng.randrange(len(random_centers))]
    centers = [(y, x), rand, (other_y, other_x)]
    return (radius, [(int(c[1]), int(c[0])) for c in centers]), None


def render(image, case, masks, arms, seed):
    variants, unavailable = {}, {}
    if "E0" in arms:
        variants["E0"] = image, None
    if "E1" in arms:
        boxed = image.copy()
        draw = ImageDraw.Draw(boxed)
        draw.rectangle(case.row["gt_boxes"][0], outline=(235, 30, 30), width=2)
        draw.rectangle(case.row["gt_boxes"][1], outline=(30, 90, 245), width=2)
        variants["E1"] = boxed, None
    if "E2" in arms:
        variants["E2"] = tint(image, masks[0] | masks[1], (150, 150, 150), 60), None
    if set(arms).intersection(MARKS):
        placement, reason = mark_centers(*masks, f"{seed}:{case.case_id}")
        if reason:
            unavailable.update({arm: reason for arm in MARKS if arm in arms})
        else:
            radius, centers = placement
            yy, xx = np.ogrid[:image.height, :image.width]
            for arm, (x, y) in zip(MARKS, centers):
                if arm in arms:
                    marker = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
                    meta = {"center_xy": [x, y], "radius_px": radius, "area_px": int(marker.sum())}
                    variants[arm] = tint(image, marker, (200, 155, 248), 88), meta
    return variants, unavailable


def review_template(case_id, image_hash):
    return {"case_id": case_id, "image_sha256": image_hash, "status": "pending",
            "author": "", "reviewer": "", "verified": "", "wrong": "", "neutral": "",
            "checks": {check: False for check in TEXT_CHECKS}, "reason": ""}


def validate_review_text(row):
    key = row["case_id"]
    texts = [row[field].strip() for field in NOTE_FIELDS.values()]
    if (not all(texts) or len(set(text.casefold() for text in texts)) != 3
            or any("\n" in text or "\r" in text or DIRECTION.search(text)
                   or re.match(r"^[AB](?:[.):]|$|\s+is\s+(?:correct|right|true)\b)", text, re.I)
                   for text in texts)):
        raise ValueError(f"{key}: review texts must be distinct one-line observations without answer cues")
    lengths = [len(re.sub(r"\s+", "", text)) for text in texts]
    if max(lengths) > 1.5 * min(lengths):
        raise ValueError(f"{key}: text controls must be within 1.5x character length")
    if not re.fullmatch(r"[0-9a-f]{64}", row["image_sha256"]):
        raise ValueError(f"{key}: expected the exported image_sha256")


def load_reviews(path, cases):
    reviews = {}
    if path is None:
        return reviews
    known = {case.case_id for case in cases}
    for row in read_jsonl(path):
        key = row["case_id"]
        if key not in known or key in reviews:
            raise ValueError(f"{path}: unknown or duplicate case_id {key}")
        status = row["status"]
        if status == "approved":
            author, reviewer = [row[key].strip().casefold() for key in ("author", "reviewer")]
            if (not author or not reviewer or author == reviewer
                    or not all(row["checks"].get(check) is True for check in TEXT_CHECKS)):
                raise ValueError(f"{key}: all review checks and two distinct human IDs are required")
            validate_review_text(row)
        elif status == "excluded":
            if not row.get("reason", "").strip():
                raise ValueError(f"{key}: missing exclusion reason")
        elif status != "pending":
            raise ValueError(f"{key}: unknown review status")
        reviews[key] = row
    return reviews


def text_inputs(review, image_sha, allow_provisional=False):
    if review is None:
        return None, "review_pending"
    if review["status"] == "excluded":
        return None, "review_excluded"
    if review["status"] == "pending":
        if not allow_provisional or not all(review.get(field, "").strip() for field in NOTE_FIELDS.values()):
            return None, "review_pending"
        validate_review_text(review)
    if review["image_sha256"] != image_sha:
        raise ValueError(f"{review['case_id']}: source image changed since export")
    return {arm: review[field].strip() for arm, field in NOTE_FIELDS.items()}, None


def model_prompt(question, options, arm, note=None):
    hints = {
        "E1": "The red box identifies the first object in the question. The blue box identifies the second object.",
        "E2": "The two queried objects have the same translucent gray overlay.",
        **dict.fromkeys(MARKS, "The image contains a translucent circular mark. Inspect the visible scene."),
    }
    prefix = hints.get(arm, "")
    if arm in NOTES:
        prefix = f"Supplementary visual observation: {note}"
    base = f"Question: {question}\nOptions:\n{options}\n"
    tail = "Briefly analyze the visual relationship, then give only the option letter inside <answer> and </answer>."
    return f"{prefix}\n{base}{tail}" if prefix else base + tail
