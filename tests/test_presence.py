# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the presence task: targets, loss normalisation, cell metrics, checkpoint args and results."""

import functools
import json
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torchvision.ops import sigmoid_focal_loss

from ultralytics.cfg import check_cfg, get_cfg
from ultralytics.data.dataset import PresenceDataset
from ultralytics.data.utils import load_dataset_cache_file
from ultralytics.engine.model import Model
from ultralytics.models.yolo.presence import PresenceResults, PresenceValidator
from ultralytics.models.yolo.presence import val as presence_val
from ultralytics.nn.tasks import PresenceModel
from ultralytics.utils.loss import v8PresenceLoss
from ultralytics.utils.metrics import PresenceMetrics
from ultralytics.utils.plotting import overlay_heatmap, plot_images
from ultralytics.utils.tal import presence_batch_targets, presence_fp_masks, presence_targets

STRIDES = [8, 16, 32]
SHAPES_64 = [(8, 8), (4, 4), (2, 2)]  # level grids of a 64x64 input


def _targets(boxes, shapes=SHAPES_64, strides=STRIDES, bs=1, batch_idx=None, **kwargs):
    """Run presence_targets on single-class boxes given in input pixels (xyxy)."""
    boxes = torch.tensor(boxes, dtype=torch.float32).view(-1, 4)
    bi = torch.zeros(len(boxes)) if batch_idx is None else torch.tensor(batch_idx, dtype=torch.float32)
    return presence_targets(boxes, torch.zeros(len(boxes)), bi, bs, shapes, strides, 1, **kwargs)


def test_targets_tiny_box_on_cell_corner():
    """A tiny box split equally over four cells marks one of them positive and ignores the other three."""
    targets, weights = _targets([[30, 30, 34, 34]])  # (32, 32) is a cell corner at strides 8, 16 and 32
    for t, w in zip(targets, weights):
        assert t.sum() == 1  # best-cell fallback
        assert (w == 0).sum() == 3  # the other cells the box touches are ignored
        assert ((w == 1) & (t == 0)).sum() == t.numel() - 4  # every other cell is a negative
        assert w[t > 0].item() == 1


@pytest.mark.parametrize("inst_pow, weight", [(0.0, 1.0), (0.5, 1 / math.sqrt(60)), (1.0, 1 / 60)])
def test_targets_large_box_and_instance_weights(inst_pow, weight):
    """Fully and mostly covered cells are positive; corner cells with a quarter of a cell covered are ignored."""
    (t,), (w,) = _targets([[4, 4, 60, 60]], shapes=[(8, 8)], strides=[8], inst_pow=inst_pow)
    assert t.sum() == 60  # 36 inner cells (score 1) + 24 edge cells (score 0.5)
    assert (w == 0).sum() == 4  # the 4 corner cells (score 0.25)
    assert torch.allclose(w[t > 0], torch.full((60,), weight))


def test_targets_shared_cell_takes_largest_weight():
    """Where two boxes mark the same cell, the cell keeps the larger of the two weights."""
    # box A marks 16 cells (weight 1/16), box B lies inside cell (3, 3) of A and marks only that cell (weight 1)
    (t,), (w,) = _targets([[0, 0, 32, 32], [24, 24, 30, 30]], shapes=[(8, 8)], strides=[8], inst_pow=1.0)
    assert t.sum() == 16
    assert w[0, 0, 3, 3] == 1
    assert torch.allclose(w[t > 0].sort().values[:15], torch.full((15,), 1 / 16))


def test_targets_empty_batch():
    """Without boxes, all targets are 0 and all weights 1."""
    targets, weights = _targets([], bs=2)
    for t, w in zip(targets, weights):
        assert t.shape[0] == 2 and t.sum() == 0 and (w == 1).all()


def test_batch_targets_cache():
    """presence_batch_targets reuses its result for the same arguments and recomputes for different ones."""
    batch = {"bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]), "cls": torch.zeros(1, 1), "batch_idx": torch.zeros(1)}
    first = presence_batch_targets(batch, 1, SHAPES_64, STRIDES, 1, tau=0.3)
    assert presence_batch_targets(batch, 1, SHAPES_64, STRIDES, 1, tau=0.3) is first
    assert presence_batch_targets(batch, 1, SHAPES_64, STRIDES, 1, tau=0.5) is not first


@pytest.fixture(scope="module")
def presence_model():
    """Return a presence model with levels at strides 8, 16 and 32."""
    return PresenceModel("yolo11n-presence.yaml", nc=1, verbose=False)


def _loss(model, logits, boxes, batch_idx, cls=None, **hyp):
    """Return the presence loss (loss * batch size) for hand-made logits and normalized xywh boxes (class 0 default)."""
    model.args = {"pres_tau": 0.3, **hyp}
    batch = {
        "bboxes": torch.tensor(boxes, dtype=torch.float32).view(-1, 4),
        "cls": torch.zeros(len(boxes), 1) if cls is None else torch.tensor(cls, dtype=torch.float32).view(-1, 1),
        "batch_idx": torch.tensor(batch_idx, dtype=torch.float32),
    }
    return v8PresenceLoss(model)(logits, batch)[0]


def test_loss_fixed_divisor_is_per_image(presence_model):
    """With pres_norm > 0 the loss of a batch is the sum of the losses of its images, whatever they contain."""
    torch.manual_seed(0)
    logits = [torch.randn(2, 1, h, w) for h, w in SHAPES_64]
    animals = [[0.25, 0.25, 0.2, 0.2], [0.7, 0.6, 0.3, 0.4]]  # both in image 0; image 1 is empty
    for norm in (10.0, 0.0):
        pair = _loss(presence_model, logits, animals, [0, 0], pres_norm=norm)
        single = _loss(presence_model, [x[:1] for x in logits], animals, [0, 0], pres_norm=norm)
        empty = _loss(presence_model, [x[1:] for x in logits], [], [], pres_norm=norm)
        # the per-batch positive count (pres_norm=0) makes the batch loss depend on the batch mix
        assert torch.allclose(pair, single + empty) == (norm > 0)


def test_loss_default_matches_positive_count_normalisation(presence_model):
    """With pres_norm=0 and boxes on cell edges (nothing ignored), the loss is the focal sum / positive cells."""
    torch.manual_seed(0)
    logits = [torch.randn(2, 1, h, w) for h, w in SHAPES_64]
    loss = _loss(presence_model, logits, [[0.25, 0.25, 0.5, 0.5], [0.75, 0.75, 0.5, 0.5]], [0, 1])
    boxes = torch.tensor([[0, 0, 32, 32], [32, 32, 64, 64]], dtype=torch.float32)  # the same boxes in pixels
    targets, weights = presence_targets(boxes, torch.zeros(2), torch.tensor([0.0, 1.0]), 2, SHAPES_64, STRIDES, 1)
    assert all((w > 0).all() for w in weights)  # nothing ignored
    focal = sum(sigmoid_focal_loss(f, t, alpha=0.25, gamma=2.0, reduction="sum") for f, t in zip(logits, targets))
    npos = sum(t.sum() for t in targets)
    assert torch.allclose(loss, focal / npos * 2)  # the criterion returns loss * batch size


def _exact_ap(scores, targets):
    """Return the step AP of unbinned scores (one threshold per distinct score), with PresenceMetrics' envelope."""
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], targets[order]
    tp, fp = np.cumsum(t), np.cumsum(1 - t)
    last = np.r_[s[1:] != s[:-1], True]  # last cell of each run of equal scores
    return PresenceMetrics._step_ap(tp[last] / t.sum(), tp[last] / (tp[last] + fp[last]))


@pytest.mark.parametrize("mean_pos, mean_neg", [(1.0, -3.0), (-12.0, -14.0)])
def test_metrics_ap_matches_exact(mean_pos, mean_neg):
    """The binned AP is within 0.002 of the exact AP, also when every score is below 0.001."""
    rng = np.random.default_rng(0)
    t = np.zeros(200_000)
    t[:2_000] = 1
    logits = np.where(t > 0, rng.normal(mean_pos, 1.0, t.size), rng.normal(mean_neg, 1.0, t.size)).astype(np.float32)
    x = torch.from_numpy(logits).view(1, 1, 400, 500)
    m = PresenceMetrics()
    m.update([x], [torch.from_numpy(t).float().view(1, 1, 400, 500)], None, x, [8])
    m.process()
    exact = _exact_ap(logits, t)
    if mean_pos < 0:
        assert x.sigmoid().max() < 0.001  # 1000 linear score bins would put every cell into the first bin
    assert exact > 0.05  # the ranking carries information, well above the positive rate of 0.01
    assert abs(m.ap - exact) < 0.002


def test_metrics_skip_ignored_cells():
    """Cells marked as not valid (ignored by the targets) are not counted."""
    x = torch.tensor([[[[3.0, -3.0], [3.0, -3.0]]]])
    t = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    valid = torch.tensor([[[[True, True], [False, True]]]])  # the confident negative at (1, 0) is ignored
    m = PresenceMetrics()
    m.update([x], [t], [valid], x, [8])
    m.process()
    assert m.hist.sum() == 3
    assert m.p == m.r == m.ap == 1.0


def test_most_accurate_map():
    """With labels the map that best covers the positive cells wins, ignored cells aside; without, the quietest."""
    t_fine, t_coarse = torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 2, 2)
    t_fine[0, 0, 1, 1] = t_coarse[0, 0, 0, 0] = 1  # image 0 has one animal, image 1 none
    v_fine, v_coarse = torch.ones_like(t_fine, dtype=torch.bool), torch.ones_like(t_coarse, dtype=torch.bool)
    v_fine[0, 0, 1, 2] = False  # an ignored cell next to the animal
    fine = torch.full_like(t_fine, 0.01)
    fine[0, 0, 1, 1] = fine[0, 0, 1, 2] = 0.9  # F1 0.88, or 0.61 if the ignored cell counted as a false alarm
    fine[1, 0, 3, 3] = 0.9  # one false alarm: mean 0.066
    coarse = torch.full_like(t_coarse, 0.05)
    coarse[0, 0, 0, 0] = 0.7  # F1 0.76
    coarse[1] = 0.03  # uniformly low: mean 0.03
    best = PresenceValidator.most_accurate_map([fine, coarse], [t_fine, t_coarse], [v_fine, v_coarse])
    assert best.tolist() == [0, 1]


def test_plot_predictions_one_tile_per_image(tmp_path, monkeypatch):
    """Validation plots write one labels and one prediction mosaic per batch, and nothing from plot_val_samples."""
    monkeypatch.setattr(presence_val, "plot_images", functools.partial(plot_images, threaded=False))  # save now
    validator = PresenceValidator(save_dir=tmp_path)
    validator.init_metrics(SimpleNamespace(names={0: "animal"}))
    batch = {
        "img": torch.rand(3, 3, 64, 64),
        "bboxes": torch.tensor([[0.25, 0.25, 0.2, 0.2]]),  # one animal in image 0, images 1 and 2 empty
        "cls": torch.zeros(1, 1),
        "batch_idx": torch.zeros(1),
        "im_file": ["a.jpg", "b.jpg", "c.jpg"],
    }
    logits = [torch.randn(3, 1, h, w) for h, w in SHAPES_64]
    preds = validator.postprocess(([x.sigmoid() for x in logits], logits))
    validator.update_metrics(preds, batch)
    validator.plot_val_samples(batch, 0)
    assert not any(tmp_path.iterdir())
    validator.plot_predictions(batch, preds, 0)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["val_batch0_labels.jpg", "val_batch0_pred.jpg"]


def test_reset_ckpt_args_keeps_presence_hyperparameters():
    """Loading a presence checkpoint keeps the pres_* arguments; other tasks drop them."""
    args = {"imgsz": 320, "data": "d.yaml", "task": "presence", "epochs": 3, "pres_tau": 0.5, "pres_conf": 0.4}
    kept = Model._reset_ckpt_args(args)
    assert kept == {"imgsz": 320, "data": "d.yaml", "task": "presence", "pres_tau": 0.5, "pres_conf": 0.4}
    assert "pres_tau" not in Model._reset_ckpt_args({**args, "task": "detect"})


def test_cfg_presence_keys():
    """The presence hyperparameters exist in the default config and are type-checked."""
    hyp = {
        "pres_norm": 20,
        "pres_alpha": 0.5,
        "pres_gamma": 1,
        "pres_inst_pow": 0.5,
        "pres_conf": 0.3,
        "pres_fp_weight": 4,
    }
    cfg = get_cfg(overrides=hyp)
    assert all(getattr(cfg, k) == v for k, v in hyp.items())
    with pytest.raises(ValueError):
        check_cfg({"pres_alpha": 1.5})


def _results():
    """Return PresenceResults of a 100x150 image letterboxed into 64x64, with two fused cells above 0.5."""
    ratio = 64 / 150
    fused = torch.full((1, 6, 8), 0.1)  # P3 grid of the 43x64 resized image: ceil(43 / 8) = 6 rows
    fused[0, 1, 2], fused[0, 5, 7] = 0.9, 0.6
    maps = [torch.full((1, 6, 8), 0.7), torch.full((1, 3, 4), 0.2), torch.full((1, 2, 2), 0.2)]
    img = np.full((100, 150, 3), 128, dtype=np.uint8)
    return PresenceResults(img, "img.jpg", {0: "animal"}, maps, [8.0, 16.0, 32.0], ratio, fused, presence_conf=0.5)


def test_results_plot_and_summary():
    """r.numpy().plot() works, and summary(), to_json() and to_df() list the flagged cells with their boxes."""
    r = _results()
    assert r.plot().shape == r.numpy().plot().shape == (100, 150, 3)
    assert isinstance(r.numpy().cpu().presence[0], torch.Tensor)
    rows = r.summary()
    assert [row["confidence"] for row in rows] == [0.9, 0.6]  # sorted by score
    assert rows[0]["box"] == {"x1": 37.5, "y1": 18.75, "x2": 56.25, "y2": 37.5}  # cells are 18.75 px in the image
    assert rows[1]["box"] == {"x1": 131.25, "y1": 93.75, "x2": 150, "y2": 100}  # clipped to the image
    assert rows[0]["level"] == "fused" and r.summary(level=0)[0]["level"] == "P3"
    assert r.summary(conf=0.95) == []
    assert r.numpy().summary() == rows
    assert len(json.loads(r.to_json())) == len(r.to_df()) == 2


def test_overlay_heatmap_keeps_image_where_score_is_zero():
    """A zero heatmap leaves the image unchanged; a heatmap of ones tints it."""
    img = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    assert np.array_equal(overlay_heatmap(img, np.zeros((5, 6), np.float16)), img)
    assert not np.array_equal(overlay_heatmap(img, torch.ones(5, 6)), img)


ANIMAL, FP_FAR, FP_NEAR = [8, 8, 24, 24], [40, 40, 56, 56], [20, 8, 30, 24]  # xyxy pixels; FP_NEAR touches ANIMAL


def _px_batch(boxes, cls, size=64):
    """Return a one-image batch of boxes given in input pixels (xyxy), with their classes (-1: false-positive)."""
    b = torch.tensor(boxes, dtype=torch.float32).view(-1, 4)
    xywh = torch.cat([(b[:, :2] + b[:, 2:]) / 2, b[:, 2:] - b[:, :2]], 1) / size
    return {"bboxes": xywh, "cls": torch.tensor(cls, dtype=torch.float32).view(-1, 1), "batch_idx": torch.zeros(len(b))}


def test_fp_masks_mark_every_touched_cell():
    """A false-positive box marks every cell it overlaps at all, on every level, but not the cells it only borders."""
    boxes = torch.tensor([[100.0, 200.0, 124.0, 220.0], [0.0, 0.0, 8.0, 8.0]])  # one box in each of two images
    masks = presence_fp_masks(boxes, torch.tensor([0.0, 1.0]), 2, [(80, 80), (40, 40), (20, 20)], STRIDES)
    assert [tuple(m.shape) for m in masks] == [(2, 1, 80, 80), (2, 1, 40, 40), (2, 1, 20, 20)]
    assert [int(m[0].sum()) for m in masks] == [12, 4, 1]  # 4x3 cells on P3, 2x2 on P4, 1 on P5
    assert [int(m[1].sum()) for m in masks] == [1, 1, 1]  # ends on a cell edge: the next cell is untouched


def test_batch_targets_fp_regions():
    """False-positive boxes are no targets; they raise the weight of the negative cells they touch, and of no other."""
    ref_targets, ref_weights = presence_batch_targets(_px_batch([ANIMAL], [0]), 1, SHAPES_64, STRIDES, 1)
    batch = _px_batch([ANIMAL, FP_FAR, FP_NEAR], [0, -1, -1])
    for fp_weight, raised in ((1.0, [0, 0, 0]), (3.0, [6, 4, 1])):
        targets, weights = presence_batch_targets(batch, 1, SHAPES_64, STRIDES, 1, fp_weight=fp_weight)
        assert all(torch.equal(t, t0) for t, t0 in zip(targets, ref_targets))
        for w, w0, t0, n in zip(weights, ref_weights, ref_targets, raised):
            changed = w != w0
            assert changed.sum() == n
            assert (w[changed] == fp_weight).all() and (w0[changed] == 1).all() and (t0[changed] == 0).all()
    # FP_NEAR touches four P3 cells: two are the animal's (positive, weight kept at 1), two are negatives beside it
    assert weights[0][0, 0, 1:3, 2].tolist() == [1.0, 1.0] and weights[0][0, 0, 1:3, 3].tolist() == [3.0, 3.0]


def test_loss_fp_weight(presence_model):
    """FP boxes change nothing at pres_fp_weight=1; otherwise their negative cells weigh more, over the same divisor."""
    torch.manual_seed(0)
    logits = [torch.randn(1, 1, h, w) for h, w in SHAPES_64]
    animal, fp = [0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.25, 0.25]  # ANIMAL and FP_FAR as normalized xywh
    base = _loss(presence_model, logits, [animal], [0])
    assert torch.allclose(_loss(presence_model, logits, [animal, fp], [0, 0], cls=[0, -1], pres_fp_weight=1.0), base)
    loss = _loss(presence_model, logits, [animal, fp], [0, 0], cls=[0, -1], pres_fp_weight=3.0)
    gt = torch.tensor([ANIMAL], dtype=torch.float32)
    targets, weights = presence_targets(gt, torch.zeros(1), torch.zeros(1), 1, SHAPES_64, STRIDES, 1, tau=0.3)
    masks = presence_fp_masks(torch.tensor([FP_FAR], dtype=torch.float32), torch.zeros(1), 1, SHAPES_64, STRIDES)
    focal = sum(
        (sigmoid_focal_loss(f, t, alpha=0.25, gamma=2.0, reduction="none") * w * (1 + 2 * (m & (t == 0)))).sum()
        for f, t, w, m in zip(logits, targets, weights, masks)
    )
    assert torch.allclose(loss, focal / sum(t.sum() for t in targets))  # batch size 1, positive count unchanged
    assert loss > base


def test_validator_ignores_fp_regions(tmp_path):
    """A false-positive box in a val batch changes neither the metrics nor the instance count."""
    torch.manual_seed(0)
    logits = [torch.randn(1, 1, h, w) for h, w in SHAPES_64]
    stats, counts = [], []
    for boxes, cls in (([ANIMAL], [0]), ([ANIMAL, FP_FAR], [0, -1])):
        validator = PresenceValidator(save_dir=tmp_path, args={"pres_fp_weight": 3.0})
        validator.init_metrics(SimpleNamespace(names={0: "animal"}))
        batch = {"img": torch.zeros(1, 3, 64, 64), **_px_batch(boxes, cls)}
        validator.update_metrics(validator.postprocess(([x.sigmoid() for x in logits], logits)), batch)
        stats.append(validator.get_stats())
        counts.append(validator.nt)
    np.testing.assert_equal(stats[1], stats[0])
    assert counts == [1, 1]


def _write_tile(img_dir, name, label=None, fp=None, label_dir=None):
    """Write a grey 64x64 jpg, and its label and false-positive label files if given, as YOLO dataset files."""
    lbl_dir = img_dir if label_dir is None else img_dir.parent / label_dir / img_dir.name
    for d in {img_dir, lbl_dir}:
        d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / f"{name}.jpg"), np.full((64, 64, 3), 128, np.uint8))
    for suffix, text in ((".txt", label), ("_fp.txt", fp)):
        if text is not None:
            (lbl_dir / f"{name}{suffix}").write_text(text)


def _presence_dataset(img_dir, label_dir=None, augment=False, **hyp):
    """Return the PresenceDataset of a folder of 64x64 images, built with the given hyperparameters."""
    data = {"names": {0: "animal"}, "channels": 3, "label_dir": label_dir}
    cfg = get_cfg(overrides=hyp)
    return PresenceDataset(img_path=str(img_dir), imgsz=64, augment=augment, hyp=cfg, data=data, task="presence")


def test_presence_dataset_reads_fp_files(tmp_path):
    """Boxes of <label>_fp.txt come out as class -1 after the labels; bad files and pres_fp_weight=1 add none."""
    img_dir = tmp_path / "train"
    _write_tile(img_dir, "a", label="0 0.25 0.25 0.25 0.25\n", fp="0 0.75 0.75 0.25 0.25 0.9\n")  # with conf column
    _write_tile(img_dir, "b", label="0 0.5 0.5 0.2 0.2\n", fp="0 0.5 0.5 0.1\n")  # malformed: ignored
    _write_tile(img_dir, "c")  # no labels at all
    a, b, c = (_presence_dataset(img_dir, pres_fp_weight=4.0)[i] for i in range(3))
    assert a["cls"].view(-1).tolist() == [0, -1]
    assert torch.allclose(a["bboxes"], torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.25, 0.25]]))
    assert b["cls"].view(-1).tolist() == [0] and len(c["cls"]) == 0
    assert all("fp_bboxes" not in lb for lb in load_dataset_cache_file(tmp_path / "train.cache")["labels"])
    assert _presence_dataset(img_dir, pres_fp_weight=1.0)[0]["cls"].view(-1).tolist() == [0]  # off: not read


def test_presence_dataset_fp_files_follow_label_dir(tmp_path):
    """With label_dir set, the false-positive file is read from beside the regular label file, not the image."""
    img_dir = tmp_path / "train"
    _write_tile(img_dir, "a", label="0 0.25 0.25 0.25 0.25\n", fp="0 0.75 0.75 0.25 0.25\n", label_dir="labels_x")
    (img_dir / "a_fp.txt").write_text("0 0.5 0.5 0.5 0.5\n")  # beside the image: not read when label_dir is set
    item = _presence_dataset(img_dir, label_dir="labels_x", pres_fp_weight=4.0)[0]
    assert item["cls"].view(-1).tolist() == [0, -1]
    assert torch.allclose(item["bboxes"][1], torch.tensor([0.75, 0.75, 0.25, 0.25]))


def test_presence_dataset_fp_boxes_follow_augmentation(tmp_path):
    """False-positive boxes go through the training augmentations like the animal boxes, here a certain flip."""
    img_dir = tmp_path / "train"
    _write_tile(img_dir, "a", label="0 0.25 0.25 0.25 0.25\n", fp="0 0.75 0.5 0.25 0.25\n")
    still = dict(mosaic=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0)
    item = _presence_dataset(img_dir, augment=True, fliplr=1.0, pres_fp_weight=4.0, **still)[0]  # always flips
    assert item["cls"].view(-1).tolist() == [0, -1]
    assert torch.allclose(item["bboxes"], torch.tensor([[0.75, 0.25, 0.25, 0.25], [0.25, 0.5, 0.25, 0.25]]))
