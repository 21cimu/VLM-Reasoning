"""CPU tests. Model generation is simulated in the end-to-end test."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

import eval_occ_e234 as cli
from evidence import (ARMS, MARKS, NOTES, TEXT_CHECKS, Case, decode_mask, digest,
                      dilate, get_masks, load_cases, load_reviews, model_prompt, read_jsonl,
                      render, text_inputs)


def jsonl(path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = Image.new("RGB", (100, 80), (25, 50, 75))
        self.source = self.root / "source.jpg"
        self.image.save(self.source)
        self.row = {"id": "insta_1_101_202", "image_id": 1, "image_path": "source.jpg",
                    "img_size": [100, 80], "question": "Where do the two rectangles meet?",
                    "options": "A. First rectangle\nB. Second rectangle", "valid_opts": ["A", "B"],
                    "gt": "A", "gt_boxes": [[20, 10, 43, 42], [43, 20, 67, 52]], "stratum": "synthetic"}
        self.dataset = self.root / "data.jsonl"
        jsonl(self.dataset, self.row)
        self.case = load_cases(self.dataset)[0]
        self.anns = {
            101: {"image_id": 1, "bbox": [20, 10, 23, 32], "segmentation": [[20, 10, 43, 10, 43, 42, 20, 42]]},
            202: {"image_id": 1, "bbox": [43, 20, 24, 32], "segmentation": [[43, 20, 67, 20, 67, 52, 43, 52]]},
        }

    def approved_review(self):
        review = cli.review_template(self.case.case_id, digest(self.source.read_bytes()))
        review.update(status="approved", author="annotator-1", reviewer="annotator-2",
                      verified="A thin vertical edge meets a horizontal stripe.",
                      wrong="A thick diagonal edge meets a horizontal stripe.",
                      neutral="The surrounding field contains a muted blue shade.")
        review["checks"] = dict.fromkeys(TEXT_CHECKS, True)
        return review

    def test_all_coco_encodings(self):
        expected = np.array([[False, True, True], [False, True, False]])
        compressed = coco_mask.encode(np.asfortranarray(expected.astype(np.uint8)))
        compressed["counts"] = compressed["counts"].decode()
        for seg in (compressed, {"size": [2, 3], "counts": [2, 3, 1]}):
            np.testing.assert_array_equal(decode_mask(seg, (3, 2)), expected)
        first, second = get_masks(self.case, self.anns, self.image.size)
        self.assertEqual(int(first.sum()), 23 * 32)
        self.assertTrue(second[20, 43])

    def test_rasters_are_label_invariant_and_controls_have_equal_area(self):
        first, second = get_masks(self.case, self.anns, self.image.size)
        variants, missing = render(self.image, self.case, (first, second), cli.VISUAL, 42)
        self.assertEqual(missing, {})
        # Flip the label, swap input mask order and reverse the raw directional suffix.
        changed = Case(dict(self.row, gt="B", id="insta_1_202_101"), self.source, self.case.pair)
        flipped, _ = render(self.image, changed, (second, first), cli.VISUAL, 42)
        self.assertEqual(self.case.case_id, changed.case_id)
        original = np.asarray(self.image)
        gray = np.any(np.asarray(variants["E2"][0]) != original, axis=-1)
        np.testing.assert_array_equal(gray, first | second)
        seam = dilate(first, 7) & dilate(second, 7)
        areas = []
        for arm in ARMS[:6]:
            np.testing.assert_array_equal(variants[arm][0], flipped[arm][0])
            if arm in MARKS:
                marked = np.any(np.asarray(variants[arm][0]) != original, axis=-1)
                areas.append(int(marked.sum()))
                if arm == MARKS[0]:
                    self.assertTrue((marked & seam).any())
                else:
                    self.assertFalse((marked & seam).any())
                if arm == MARKS[2]:
                    self.assertFalse((marked & dilate(first | second, 7)).any())
                self.assertEqual(int(marked.sum()), variants[arm][1]["area_px"])
        self.assertEqual(len(set(areas)), 1)
        self.assertEqual(len({model_prompt("Q", "O", arm) for arm in MARKS}), 1)
        full = np.ones_like(first)
        _, missing = render(self.image, self.case, (full, full), cli.VISUAL, 42)
        self.assertEqual(set(missing), set(MARKS))

    def test_blind_export_and_human_review_gates(self):
        packet = self.root / "blind"
        cli.blind_export([self.case], packet, 42)
        blind = next(read_jsonl(packet / "cases.jsonl"))
        self.assertEqual(set(blind), {"case_id", "question", "image"})
        self.assertNotIn(self.row["id"], (packet / "cases.jsonl").read_text())
        self.assertEqual(text_inputs(next(read_jsonl(packet / "reviews.jsonl")), "")[1], "review_pending")
        approvals = self.root / "approved.jsonl"
        approved = self.approved_review()
        jsonl(approvals, approved)
        review = load_reviews(approvals, [self.case])[self.case.case_id]
        self.assertEqual(set(text_inputs(review, approved["image_sha256"])[0]), set(NOTES))
        with self.assertRaisesRegex(ValueError, "source image changed"):
            text_inputs(review, "another-image")
        for bad in (dict(approved, verified="The first rectangle is in front of the second."),
                    dict(approved, verified="B is correct given the visible vertical edge."),
                    dict(approved, neutral="Blue."),
                    dict(approved, reviewer="ANNOTATOR-1"),
                    dict(approved, checks=dict(approved["checks"], image_only=False))):
            jsonl(approvals, bad)
            with self.assertRaises(ValueError):
                load_reviews(approvals, [self.case])

    def test_mock_inference_restart_and_scoring(self):
        review_path = self.root / "pending.jsonl"
        pending = dict(self.approved_review(), status="pending", author="", reviewer="",
                       checks=dict.fromkeys(TEXT_CHECKS, False))
        jsonl(review_path, pending)
        reviews = load_reviews(review_path, [self.case])
        runs = self.root / "preds.jsonl"
        args = SimpleNamespace(arms=["E0", "E1", "E2", "E3", "E4"], reviews=review_path,
                               out=runs, model="treevgr", study=self.root, seed=42,
                               allow_provisional_reviews=True)
        calls = []

        class MockRunner:
            def __init__(self, model, model_root):
                pass

            def infer(self, images, prompt):
                calls.append(prompt)
                if len(calls) == 2:
                    raise RuntimeError("simulated interruption")
                self_test.assertEqual(len(images), 1)
                letter = "A" if "translucent gray" in prompt else "B"
                return f"<answer>{letter}</answer>", list(images[0].size)

        self_test = self
        fake_runner = (MockRunner, lambda raw, opts: raw[8], lambda raw: [])
        with patch.object(cli, "run_protocol", return_value={"checkpoint": str(self.root / "fake")}), \
             patch.object(cli, "import_runner", return_value=fake_runner):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                cli.run([self.case], self.anns, reviews, args)
            with runs.open("ab") as stream:
                stream.write(b'{"partial')
            cli.run([self.case], self.anns, reviews, args)
            cli.run([self.case], self.anns, reviews, args)
            self.assertEqual(len(calls), len(ARMS) + 1)
            with self.assertRaisesRegex(ValueError, "use a fresh --out"):
                formal = SimpleNamespace(**(vars(args) | {"allow_provisional_reviews": False}))
                cli.run([self.case], self.anns, reviews, formal)

        records = list(read_jsonl(runs))
        self.assertEqual(len(records), len(ARMS))
        self.assertTrue(all(self.row["id"] not in row["prompt"] for row in records))
        scorefile = self.root / "score.json"
        cli.score([self.case], SimpleNamespace(model="treevgr", predictions=[runs],
                  treevgr_e0=None, out=scorefile))
        scoring = json.loads(scorefile.read_text(encoding="utf-8"))
        self.assertEqual(scoring["e4_review_mode"], "provisional")
        self.assertEqual({row["e4_review_mode"] for row in records if row["condition"] in NOTES},
                         {"provisional"})
        report = scoring["cohorts"]["all"]
        result = report["families"]["E2"]["comparisons"][0]
        self.assertEqual((result["n_pairs"], result["wrong_to_right"], result["right_to_wrong"]), (1, 1, 0))
        self.assertEqual(result["correction_rate"], 1.0)
        self.assertIsNone(result["harm_rate"])
        self.assertEqual(report["families"]["E4"]["common_n"], 1)
        records[-1]["e4_review_mode"] = "human_reviewed"
        jsonl(runs, *records)
        with self.assertRaisesRegex(ValueError, "same review file and review mode"):
            cli.score([self.case], SimpleNamespace(model="treevgr", predictions=[runs],
                      treevgr_e0=None, out=self.root / "mixed.json"))

    def test_provisional_requires_complete_valid_draft_bound_to_image(self):
        pending = dict(self.approved_review(), status="pending", author="", reviewer="",
                       checks=dict.fromkeys(TEXT_CHECKS, False))
        sha = pending["image_sha256"]
        self.assertEqual(text_inputs(pending, sha), (None, "review_pending"))
        self.assertEqual(set(text_inputs(pending, sha, allow_provisional=True)[0]), set(NOTES))
        self.assertEqual(text_inputs(dict(pending, wrong=""), sha, True), (None, "review_pending"))
        self.assertEqual(text_inputs(dict(pending, status="excluded"), sha, True),
                         (None, "review_excluded"))
        with self.assertRaisesRegex(ValueError, "source image changed"):
            text_inputs(pending, "changed", True)
        with self.assertRaisesRegex(ValueError, "without answer cues"):
            text_inputs(dict(pending, verified="B is correct given the visible vertical edge."), sha, True)

    def test_historical_cohort_merges_shards_and_rejects_overlap(self):
        rows = [
            self.row,
            dict(self.row, id="insta_2_303_404", image_id=2, gt="B"),
        ]
        jsonl(self.dataset, *rows)
        cases = load_cases(self.dataset)
        paths = (self.root / "pilot_s0of2.jsonl", self.root / "pilot_s1of2.jsonl")
        for path, row in zip(paths, rows):
            jsonl(path, dict(row, condition="A_full", dataset="InstaOrder-Occ",
                             pred_letter="A", correct=int(row["gt"] == "A"),
                             pred_boxes=row["gt_boxes"], proc_wh=row["img_size"]))
        matcher = SimpleNamespace(matched_1to1_iou05=lambda boxes, gt, *_: (boxes == gt, 1.0))
        with patch.dict(sys.modules, {"analyze_blue2_pilot": matcher}), patch.object(sys, "path", sys.path.copy()):
            groups, count = cli.historical_cohorts(paths, self.root, cases)
            self.assertEqual(count, 2)
            self.assertEqual(groups["historical_treevgr_both_boxes"], {r["id"] for r in rows})
            self.assertEqual(groups["historical_treevgr_both_boxes_wrong"], {rows[1]["id"]})
            with self.assertRaisesRegex(ValueError, "unknown or duplicate"):
                cli.historical_cohorts([paths[0], paths[0]], self.root, cases)

    def test_paired_clustering(self):
        ids = list(range(4))
        records = {(case, arm): {"pred_letter": "A", "correct": int(arm == "E2")}
                   for case in ids for arm in ("E0", "E2")}
        same_image = dict.fromkeys(ids, "one-image")
        row = cli.paired(ids, "E0", "E2", records, same_image)
        self.assertEqual(row["mcnemar_exact_p"], 0.125)
        self.assertEqual(row["image_cluster_exact_p"], 1.0)
        independent = cli.paired(ids, "E0", "E2", records, dict(zip(ids, ids)))
        self.assertEqual(independent["image_cluster_exact_p"], 0.125)


if __name__ == "__main__":
    unittest.main()
