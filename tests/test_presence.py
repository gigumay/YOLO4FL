# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the presence task: targets, loss normalisation, cell metrics, checkpoint args and results."""

import json
import math

import numpy as np
import pytest
import torch
from torchvision.ops import sigmoid_focal_loss

from ultralytics.cfg import check_cfg, get_cfg
from ultralytics.engine.model import Model
from ultralytics.models.yolo.presence import PresenceResults
from ultralytics.nn.tasks import PresenceModel
from ultralytics.utils.loss import v8PresenceLoss
from ultralytics.utils.metrics import PresenceMetrics
from ultralytics.utils.plotting import overlay_heatmap
from ultralytics.utils.tal import presence_batch_targets, presence_targets

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


def _loss(model, logits, boxes, batch_idx, **hyp):
    """Return the presence loss (loss * batch size) for hand-made logits and normalized xywh boxes."""
    model.args = {"pres_tau": 0.3, **hyp}
    batch = {
        "bboxes": torch.tensor(boxes, dtype=torch.float32).view(-1, 4),
        "cls": torch.zeros(len(boxes), 1),
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


def test_reset_ckpt_args_keeps_presence_hyperparameters():
    """Loading a presence checkpoint keeps the pres_* arguments; other tasks drop them."""
    args = {"imgsz": 320, "data": "d.yaml", "task": "presence", "epochs": 3, "pres_tau": 0.5, "pres_conf": 0.4}
    kept = Model._reset_ckpt_args(args)
    assert kept == {"imgsz": 320, "data": "d.yaml", "task": "presence", "pres_tau": 0.5, "pres_conf": 0.4}
    assert "pres_tau" not in Model._reset_ckpt_args({**args, "task": "detect"})


def test_cfg_presence_keys():
    """The presence hyperparameters exist in the default config and are type-checked."""
    hyp = {"pres_norm": 20, "pres_alpha": 0.5, "pres_gamma": 1, "pres_inst_pow": 0.5, "pres_conf": 0.3}
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
